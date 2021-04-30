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


class MDDNet(nn.Module):
    def __init__(self, base_net='ResNet50', use_bottleneck=True, bottleneck_dim=1024, width=1024, class_num=31, lambda_=5e-2):
        super(MDDNet, self).__init__()
        ## set base network
        self.base_network = backbone.network_dict[base_net]()
        self.use_bottleneck = use_bottleneck
        #self.grl_layer = GradientReverseLayer()
        self.grl_layer = GradientReversal(lambda_=lambda_)
        
        self.bottleneck_layer_list = [nn.Linear(self.base_network.output_num(), bottleneck_dim), nn.BatchNorm1d(bottleneck_dim), nn.ReLU(), nn.Dropout(0.5)]
        self.bottleneck_layer = nn.Sequential(*self.bottleneck_layer_list)
        
        self.classifier_layer_list = [nn.Linear(bottleneck_dim, width), nn.BatchNorm1d(width), nn.ReLU(), nn.Linear(width, class_num)]
        self.classifier_layer = nn.Sequential(*self.classifier_layer_list)
        
        self.classifier_layer_2_list = [nn.Linear(bottleneck_dim, width), nn.BatchNorm1d(width), nn.ReLU(), nn.Linear(width, class_num)]
        self.classifier_layer_2 = nn.Sequential(*self.classifier_layer_2_list)
        
        self.softmax = nn.Softmax(dim=1)

        ## initialization
        self.bottleneck_layer[0].weight.data.normal_(0, 0.005)
        self.bottleneck_layer[0].bias.data.fill_(0.1)
        for dep in range(2):
            self.classifier_layer_2[dep * 3].weight.data.normal_(0, 0.01)
            self.classifier_layer_2[dep * 3].bias.data.fill_(0.0)
            self.classifier_layer[dep * 3].weight.data.normal_(0, 0.01)
            self.classifier_layer[dep * 3].bias.data.fill_(0.0)


        ## collect parameters
        #self.parameter_list = [{"params":self.base_network.parameters(), "lr":0.1},
        #                       {"params":self.bottleneck_layer.parameters(), "lr":1},
        #                       {"params":self.classifier_layer.parameters(), "lr":1},
        #                       {"params":self.classifier_layer_2.parameters(), "lr":1}]
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
    def __init__(self, base_net='ResNet50', width=1024, class_num=31, use_bottleneck=True, use_gpu=True, srcweight=3, lambda_=5e-2):
        self.c_net = MDDNet(base_net, use_bottleneck, width, width, class_num, lambda_=lambda_)
        self.c_net = nn.DataParallel(self.c_net)

        self.use_gpu = use_gpu
        self.is_train = False
        self.iter_num = 0
        self.class_num = class_num
        if self.use_gpu:
            self.c_net = self.c_net.cuda()
        self.srcweight = srcweight


    def get_loss(self, inputs, labels_source):
        class_criterion = nn.CrossEntropyLoss()

        _, outputs, _, outputs_adv = self.c_net(inputs)

        classifier_loss = class_criterion(outputs[:labels_source.size(0)], labels_source)

        target_adv = outputs.max(1)[1]
        #print(target_adv)
        target_adv_src = target_adv[:labels_source.size(0)]
        target_adv_tgt = target_adv[labels_source.size(0):]

        classifier_loss_adv_src = class_criterion(outputs_adv[:labels_source.size(0)], target_adv_src)

        #logloss_tgt = torch.log(1 - F.softmax(outputs_adv[labels_source.size(0):], dim=1))
        # According to issue on github
        logloss_tgt = torch.log(torch.clamp(1 - F.softmax(outputs_adv[labels_source.size(0):], dim=1), min=1e-15))
        classifier_loss_adv_tgt = F.nll_loss(logloss_tgt, target_adv_tgt)
        #loss_tgt = 1 - F.softmax(outputs_adv[labels_source.size(0):], dim=1)
        #classifier_loss_adv_tgt = F.nll_loss(loss_tgt, target_adv_tgt)

        #transfer_loss = self.srcweight*classifier_loss_adv_src + classifier_loss_adv_tgt
        transfer_loss = self.srcweight*classifier_loss_adv_src + classifier_loss_adv_tgt
        #print('\t classifier_loss_adv_src = ',classifier_loss_adv_src.cpu().item(), ' ; classifier_loss_adv_tgt = ', classifier_loss_adv_tgt.cpu().item())
        self.iter_num += 1

        return classifier_loss, transfer_loss

    def predict(self, inputs):
        #_, _, softmax_outputs,_= self.c_net(inputs)
        #return softmax_outputs
        _, _, _ , outputs_adv= self.c_net(inputs)
        return F.softmax(outputs_adv, dim=1)

    def get_parameter_list(self):
        #return self.c_net.parameter_list
        return self.c_net.parameters()

    def set_train(self, mode):
        self.c_net.train(mode)
        self.is_train = mode
