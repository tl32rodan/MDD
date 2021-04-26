import tqdm
import os
import os.path as osp
import argparse
from utils.config import Config
from torch.autograd import Variable
import torch
from sklearn.metrics import confusion_matrix
import numpy as np


class INVScheduler(object):
    def __init__(self, gamma, decay_rate, init_lr=0.001):
        self.gamma = gamma
        self.decay_rate = decay_rate
        self.init_lr = init_lr

    def next_optimizer(self, group_ratios, optimizer, num_iter):
        lr = self.init_lr * (1 + self.gamma * num_iter) ** (-self.decay_rate)
        i=0
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr * group_ratios[i]
            i+=1
        return optimizer


#==============eval
def evaluate(model_instance, input_loader, num_classes=12):
    ori_train_state = model_instance.is_train
    model_instance.set_train(False)
    num_iter = len(input_loader)
    iter_test = iter(input_loader)
    first_test = True
    label_indices = np.arange(num_classes)

    for i in range(num_iter):
        data = iter_test.next()
        inputs = data[0]
        labels = data[1]
        if model_instance.use_gpu:
            inputs = Variable(inputs.cuda())
            labels = Variable(labels.cuda())
        else:
            inputs = Variable(inputs)
            labels = Variable(labels)
        probabilities = model_instance.predict(inputs)

        probabilities = probabilities.data.float()
        labels = labels.data.float()
        if first_test:
            all_probs = probabilities
            all_labels = labels
            first_test = False
        else:
            all_probs = torch.cat((all_probs, probabilities), 0)
            all_labels = torch.cat((all_labels, labels), 0)

    _, predict = torch.max(all_probs, 1)
    cmx = confusion_matrix(all_labels.cpu(), predict.cpu(), labels=label_indices)
    print(cmx)
    c = np.diag(cmx)/np.sum(cmx, 1)
    print(c, np.sum(c)/num_classes)
    accuracy = torch.sum(torch.squeeze(predict) == all_labels) / float(all_labels.size()[0])

    model_instance.set_train(ori_train_state)
    return {'accuracy':accuracy}

def train(model_instance, train_source_loader, train_target_loader, test_source_loader, test_target_loader,
          group_ratios, max_iter, optimizer, lr_scheduler, eval_interval, num_classes=12):
    model_instance.set_train(True)
    print("start train...")
    iter_num = 0
    epoch = 0
    total_progress_bar = tqdm.tqdm(desc='Train iter', total=max_iter)
    while True:
        for (datas, datat) in tqdm.tqdm(
                zip(train_source_loader, train_target_loader),
                total=min(len(train_source_loader), len(train_target_loader)),
                desc='Train epoch = {}'.format(epoch), ncols=80, leave=False):
            inputs_source, labels_source = datas
            inputs_target, labels_target = datat

            optimizer = lr_scheduler.next_optimizer(group_ratios, optimizer, iter_num/5)
            optimizer.zero_grad()

            if model_instance.use_gpu:
                inputs_source, inputs_target, labels_source = Variable(inputs_source).cuda(), Variable(
                    inputs_target).cuda(), Variable(labels_source).cuda()
            else:
                inputs_source, inputs_target, labels_source = Variable(inputs_source), Variable(
                    inputs_target), Variable(labels_source)

            train_batch(model_instance, inputs_source, labels_source, inputs_target, optimizer)

            # val
            if iter_num % eval_interval == 0 and iter_num != 0:
                print("===================================")
                print("Stest")
                eval_result = evaluate(model_instance, test_source_loader, num_classes=num_classes)
                print(eval_result['accuracy'].item())
                print("-----------------------------------")
                print("Ttest")
                eval_result = evaluate(model_instance, test_target_loader, num_classes=num_classes)
                print(eval_result['accuracy'].item())
                torch.save(model_instance.c_net.state_dict(), osp.join(args.save, str(int(eval_result['accuracy'].item()))+'.pth'))
            iter_num += 1
            total_progress_bar.update(1)
        epoch += 1
        if iter_num >= max_iter:
            break
    print('finish train')

def train_batch(model_instance, inputs_source, labels_source, inputs_target, optimizer):
    inputs = torch.cat((inputs_source, inputs_target), dim=0)
    total_loss = model_instance.get_loss(inputs, labels_source)
    total_loss.backward()
    optimizer.step()

if __name__ == '__main__':
    from model.MDD import MDD
    from preprocess.data_provider import load_images

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='all sets of configuration parameters',
                        default='../config/dann.yml')
    parser.add_argument('--dataset', default='VIS_dataset', type=str,
                        help='which dataset')
    parser.add_argument('--root_folder', default='./data', type=str,
                        help='root folder of dataset')
    parser.add_argument('--src_address', default=None, type=str,
                        help='address of image list of source dataset')
    parser.add_argument('--tgt_address', default=None, type=str,
                        help='address of image list of target dataset')
    parser.add_argument('--src_test_address', default=None, type=str,
                        help='address of image list of source testing dataset')
    parser.add_argument('--tgt_test_address', default=None, type=str,
                        help='address of image list of target testing dataset')
    parser.add_argument('--save', default='./save/0000_0000', type=str,
                        help='checkpoint saving directory')
    parser.add_argument('--num_classes', default=8, type=int,
                        help='number of classes')
    parser.add_argument('--batch_size', default=32, type=int,
                        help='Batch size')
    args = parser.parse_args()

    if not osp.exists(args.save):
        os.system('mkdir -p '+args.save)

    cfg = Config(args.config)
    source_file = osp.join(args.root_folder, args.src_address)
    target_file = osp.join(args.root_folder, args.tgt_address)
    source_test_file = osp.join(args.root_folder, args.src_test_address)
    target_test_file = osp.join(args.root_folder, args.tgt_test_address)


    if args.dataset == 'Office-31':
        class_num = 31
        width = 1024
        srcweight = 4
        is_cen = False
        resize_size = 256
        crop_size = 224
    elif args.dataset == 'VisDA':
        class_num = 12
        width = 1024
        srcweight = 3
        is_cen = False
        resize_size = 256
        crop_size = 224
    elif args.dataset == 'Office-Home':
        class_num = 65
        width = 2048
        srcweight = 2
        is_cen = False
        resize_size = 256
        crop_size = 224
        # Another choice for Office-home:
        # width = 1024
        # srcweight = 3
        # is_cen = True
    elif args.dataset == 'VIS_dataset':
        class_num = args.num_classes
        width = 2048
        srcweight = 2
        is_cen = True
        resize_size = 400
        crop_size = 400
    else:
        width = -1

    model_instance = MDD(base_net='ResNet50', width=width, use_gpu=True, class_num=class_num, srcweight=srcweight)

    train_source_loader = load_images(source_file, batch_size=args.batch_size, resize_size=resize_size, crop_size=crop_size, is_cen=is_cen, root_folder=args.root_folder)
    train_target_loader = load_images(target_file, batch_size=args.batch_size, resize_size=resize_size, crop_size=crop_size, is_cen=is_cen, root_folder=args.root_folder)
    test_source_loader = load_images(source_test_file, batch_size=4,  resize_size=resize_size, crop_size=crop_size, is_train=False, is_cen=is_cen, root_folder=args.root_folder)
    test_target_loader = load_images(target_test_file, batch_size=4,  resize_size=resize_size, crop_size=crop_size, is_train=False, is_cen=is_cen, root_folder=args.root_folder)

    param_groups = model_instance.get_parameter_list()
    group_ratios = [group['lr'] for group in param_groups]


    assert cfg.optim.type == 'sgd', 'Optimizer type not supported!'

    optimizer = torch.optim.SGD(param_groups, **cfg.optim.params)

    assert cfg.lr_scheduler.type == 'inv', 'Scheduler type not supported!'
    lr_scheduler = INVScheduler(gamma=cfg.lr_scheduler.gamma,
                                decay_rate=cfg.lr_scheduler.decay_rate,
                                init_lr=cfg.init_lr)

    train(model_instance, train_source_loader, train_target_loader, test_source_loader, test_target_loader, group_ratios,
          max_iter=100000, optimizer=optimizer, lr_scheduler=lr_scheduler, eval_interval=1000, num_classes=class_num)

