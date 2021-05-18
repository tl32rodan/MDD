import torch.nn as nn
import model.backbone as backbone
import torch.nn.functional as F
import torch
import numpy as np

class GradientReversalFunction(torch.autograd.Function):
    """
    Gradient Reversal Layer from:
    Unsupervised Domain Adaptation by Backpropagation (Ganin & Lempitsky, 2015)
    Forward pass is the identity function. In the backward pass,
    the upstream gradients are multiplied by -lambda (i.e. gradient is reversed)
    """

    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grads):
        lambda_ = ctx.lambda_
        lambda_ = grads.new_tensor(lambda_)
        dx = -lambda_ * grads
        return dx, None
class GradientReversal(torch.nn.Module):
    def __init__(self, lambda_=1):
        super(GradientReversal, self).__init__()
        self.lambda_ = lambda_

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)
        #return GradientReverseLayer()(x)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.01)
        m.bias.data.normal_(0.0, 0.01)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.01)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.01)
        m.bias.data.normal_(0.0, 0.01)


class ResClassifier(nn.Module):
    def __init__(self, num_classes=12,num_layer = 2,num_unit=2048,prob=0.5,middle=1000):
        super(ResClassifier, self).__init__()
        layers = []
        # currently 10000 units
        layers.append(nn.Dropout(p=prob))
        layers.append(nn.Linear(num_unit,middle))
        layers.append(nn.BatchNorm1d(middle,affine=True))
        layers.append(nn.ReLU(inplace=True))

        for i in range(num_layer-1):
            layers.append(nn.Dropout(p=prob))
            layers.append(nn.Linear(middle,middle))
            layers.append(nn.BatchNorm1d(middle,affine=True))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(middle,num_classes))
        self.classifier = nn.Sequential(*layers)

    def set_lambda(self, lambd):
        self.lambd = lambd
    def forward(self, x):
        x = self.classifier(x)
        return x


class MDDNet(nn.Module):
    def __init__(self, base_net='ResNet50', use_bottleneck=True, bottleneck_dim=1024, width=1024, class_num=31, \
                lambda_=5e-2):
        super(MDDNet, self).__init__()
        ## set base network
        self.base_network = backbone.network_dict[base_net]()
        self.use_bottleneck = use_bottleneck
        #self.grl_layer = GradientReverseLayer()
        self.grl_layer = GradientReversal(lambda_=lambda_)
        
        self.bottleneck_layer_list = [nn.Linear(self.base_network.output_num(), bottleneck_dim), nn.BatchNorm1d(bottleneck_dim), nn.ReLU(), nn.Dropout(0.5)]
        self.bottleneck_layer = nn.Sequential(*self.bottleneck_layer_list)
        
        self.classifier_layer = ResClassifier(num_classes=class_num, num_unit=bottleneck_dim)
        self.classifier_layer_2 = ResClassifier(num_classes=class_num, num_unit=bottleneck_dim)
        
        self.softmax = nn.Softmax(dim=1)

        ## initialization
        self.bottleneck_layer.apply(weights_init)
        self.classifier_layer.apply(weights_init)
        self.classifier_layer_2.apply(weights_init)

        ## collect parameters
        self.parameter_dict = {"G": self.base_network.parameters(),
                               "Bottle": self.bottleneck_layer.parameters(),
                               "F": self.classifier_layer.parameters(),
                               "F_Prime": self.classifier_layer_2.parameters()}
    def forward(self, inputs):
        features = self.base_network(inputs)
        if self.use_bottleneck:
            features = self.bottleneck_layer(features)
        #features_adv = self.grl_layer.apply(features)
        features_adv = self.grl_layer(features)
        outputs_adv = self.classifier_layer_2(features_adv)
        
        outputs = self.classifier_layer(features)
        softmax_outputs = self.softmax(outputs)

        return features, outputs, softmax_outputs, outputs_adv

class MDD(object):
    def __init__(self, base_net='ResNet50', width=1024, class_num=31, use_bottleneck=True, use_gpu=True, \
                 srcweight=3, lambda_=5e-2):
        self.c_net = MDDNet(base_net, use_bottleneck, width, width, class_num, lambda_=lambda_)
        if use_gpu:
            self.c_net = nn.DataParallel(self.c_net)

        self.use_gpu = use_gpu
        self.is_train = False
        self.iter_num = 0
        self.class_num = class_num
        if self.use_gpu:
            self.c_net = self.c_net.cuda()
        self.srcweight = srcweight

    def get_loss(self, inputs, cls_gt, len_source=-1):
        self.iter_num += 1
        class_criterion = nn.CrossEntropyLoss()

        if len_source == -1:
            len_source = cls_gt.size(0)
        _, outputs, _, outputs_adv = self.c_net(inputs)

        classifier_loss = class_criterion(outputs[:cls_gt.size(0)], cls_gt)

        target_adv = outputs.max(1)[1]
        #print(target_adv)
        target_adv_src = target_adv[:len_source]
        target_adv_tgt = target_adv[len_source:]

        classifier_loss_adv_src = class_criterion(outputs_adv[:len_source], target_adv_src)

        #logloss_tgt = torch.log(1 - F.softmax(outputs_adv[cls_gt.size(0):], dim=1))
        # According to issue on github
        logloss_tgt = torch.log(torch.clamp(1 - F.softmax(outputs_adv[len_source:], dim=1), min=1e-15))
        classifier_loss_adv_tgt = F.nll_loss(logloss_tgt, target_adv_tgt)
        #loss_tgt = 1 - F.softmax(outputs_adv[cls_gt.size(0):], dim=1)
        #classifier_loss_adv_tgt = F.nll_loss(loss_tgt, target_adv_tgt)

        #transfer_loss = self.srcweight*classifier_loss_adv_src + classifier_loss_adv_tgt
        transfer_loss = self.srcweight*classifier_loss_adv_src + classifier_loss_adv_tgt
        #print('\t classifier_loss_adv_src = ',classifier_loss_adv_src.cpu().item(), ' ; classifier_loss_adv_tgt = ', classifier_loss_adv_tgt.cpu().item())
        return classifier_loss, transfer_loss
            

    def predict(self, inputs):
        _, _, softmax_outputs,_= self.c_net(inputs)
        return softmax_outputs
        #_, _, _ , outputs_adv= self.c_net(inputs)
        #return F.softmax(outputs_adv, dim=1)

    def get_parameter_dict(self):
        if self.use_gpu:
            return self.c_net.module.parameter_dict
        else:
            return self.c_net.parameter_dict
        #return self.c_net.parameters()

    def set_train(self, mode):
        self.c_net.train(mode)
        self.is_train = mode
