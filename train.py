import tqdm
import os
import os.path as osp
import argparse
from utils.config import Config
from torch.autograd import Variable
import torch
from sklearn.metrics import confusion_matrix
import numpy as np
from preprocess.data_list import ImageList


#==============eval
def evaluate(model_instance, input_loader, num_classes=12, max_iter=None):
    np.set_printoptions(linewidth=200)
    ori_train_state = model_instance.is_train
    model_instance.set_train(False)
    num_iter = len(input_loader) if max_iter is None else max_iter
    iter_test = iter(input_loader)
    first_test = True
    label_indices = np.arange(num_classes)

    for i in range(num_iter):
        data = iter_test.next()
        inputs = data[0]
        labels = data[1]
        if model_instance.use_gpu:
            inputs = inputs.cuda()
            labels = labels.cuda()
        
        probabilities = model_instance.predict(inputs)
        probabilities = probabilities.data.float()
        #labels = labels.data.float()
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
    print(c)
    print('avg_cls_acc = ', np.sum(c)/num_classes)
    accuracy = torch.sum(torch.squeeze(predict) == all_labels).float() / float(all_labels.size()[0])
    print('overall_acc = ', accuracy.item())

    model_instance.set_train(ori_train_state)
    return {'accuracy':accuracy}

def train(args, model_instance, train_source_loader, train_target_loader, test_source_loader, test_target_loader,
          #group_ratios, max_iter, optimizer, lr_scheduler, eval_interval, num_classes=12):
          max_iter, eval_interval, num_classes=12, use_ssda=False, train_labeled_target_loader=None,\
          true_weights=None, source_weight=None):
    model_instance.set_train(True)
    print("start train...")

    optimizer = torch.optim.SGD(list(model_instance.c_net.parameters()), lr=0.001, momentum=0.9 ,weight_decay=0.0005, nesterov=True)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 10, 0.8)
    iter_num = 0
    epoch = 0
    total_progress_bar = tqdm.tqdm(desc='Train iter', total=max_iter)

    while True:
        if use_ssda:
            all_loaders = zip(train_source_loader, train_labeled_target_loader, train_target_loader)
            total_len = min(len(train_source_loader), len(train_target_loader), len(train_labeled_target_loader))
        else:
            all_loaders = zip(train_source_loader, train_target_loader)
            total_len = min(len(train_source_loader), len(train_target_loader))

        for data_tuple in tqdm.tqdm(
                all_loaders,
                total=total_len,
                desc='Train epoch = {}'.format(epoch), ncols=80, leave=False):
            model_instance.set_train(True)
            
            if use_ssda:
                datas, datat_l, datat = data_tuple
                inputs_target_l, labels_target_l = datat_l
            else:
                datas, datat = data_tuple

            inputs_source, labels_source = datas
            inputs_target, _ = datat

            #optimizer = lr_scheduler.next_optimizer(group_ratios, optimizer, iter_num/5)
            optimizer.zero_grad()
            
            if use_ssda:
                inputs = torch.cat((inputs_source, inputs_target_l, inputs_target), dim=0)
                cls_gt = torch.cat((labels_source, labels_target_l), dim=0)
            else:
                inputs = torch.cat((inputs_source, inputs_target), dim=0)
                cls_gt = labels_source

            if model_instance.use_gpu:
                inputs, cls_gt = inputs.cuda(), cls_gt.cuda()
            
            cls_loss, dis_loss = train_batch(model_instance, inputs, cls_gt, optimizer, len(inputs_source),\
                                             args.use_oracle_IW, true_weights, source_weight)
            #if cls_loss > 10 or dis_loss > 10:
            #    print('Classifier loss = ', cls_loss, ' ; Discrepency loss = ', dis_loss)
            #print('Classifier loss = ', cls_loss, ' ; Discrepency loss = ', dis_loss)
            #print('lr = ', optimizer.param_groups[0]['lr'])
            # val
            if iter_num % eval_interval == 0 and iter_num != 0:
                print("===================================")
                print('lr = ', optimizer.param_groups[0]['lr'])
                print('Classifier loss = ', cls_loss, ' ; Discrepency loss = ', dis_loss)
                print("Stest")
                eval_result = evaluate(model_instance, test_source_loader, num_classes=num_classes)
                print("-----------------------------------")
                print("Ttest")
                eval_result = evaluate(model_instance, test_target_loader, num_classes=num_classes)
                torch.save(model_instance.c_net.state_dict(), osp.join(args.save, str(int(eval_result['accuracy'].item()*100))+'.pth'))

                lr_scheduler.step()

            iter_num += 1
            total_progress_bar.update(1)
        epoch += 1
        if iter_num >= max_iter:
            break
    print('finish train')

def train_batch(model_instance, inputs, cls_gt, optimizer, len_source=-1, \
                use_oracle_IW=False, true_weights=None, source_weight=None):
    classifier_loss, discrepency_loss = model_instance.get_loss(inputs, cls_gt, len_source,\
                                                                use_oracle_IW, true_weights, source_weight)
    total_loss = classifier_loss + discrepency_loss
    total_loss.backward()
    optimizer.step()

    return classifier_loss.cpu().item(), discrepency_loss.cpu().item()

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
    parser.add_argument('--labeled_tgt_address', default=None, type=str,
                        help='For Semi DA ; address of labeled image list of target dataset')
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
    parser.add_argument('--labeled_tgt_batch_size', default=8, type=int,
                        help='Batch size of labeled target data in a minibatch')
    parser.add_argument('--lambda_', default=5e-2, type=float,
                        help='coefficient of gradient passed by GradientReversal')
    parser.add_argument('--srcweight', default=None, type=int,
                        help='source weight in MDD')
    parser.add_argument('--reuse_ckpt', default=False, type=bool,
                        help='To reuse checkpoint or not')
    parser.add_argument('--use_oracle_IW', default=False, action='store_true',
                        help='Use oracle importance weight or not')
    parser.add_argument('--reuse_ckpt_weight', default=None, type=int,
                        help='The weight to use')
    args = parser.parse_args()

    if not osp.exists(args.save):
        os.system('mkdir -p '+args.save)
    
    use_ssda = not (args.labeled_tgt_address is None)
    print("Use_SSDA = ", use_ssda)

    cfg = Config(args.config)
    source_file = osp.join(args.root_folder, args.src_address)
    target_file = osp.join(args.root_folder, args.tgt_address)
    if use_ssda:
        labeled_target_file = osp.join(args.root_folder, args.labeled_tgt_address)
    if args.src_test_address is None:
        args.src_test_address = args.src_address
    if args.tgt_test_address is None:
        args.tgt_test_address = args.tgt_address
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
        srcweight = 2
        is_cen = False
        resize_size = 256
        crop_size = 224
        eval_interval=3000
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
        eval_interval=500
    else:
        width = -1
    # Custom srcweight
    if not (args.srcweight is None):
        srcweight = args.srcweight

    # Create model
    model_instance = MDD(base_net='ResNet101', width=width, use_gpu=True, class_num=class_num, \
                         srcweight=srcweight, lambda_=args.lambda_)
   
    if args.reuse_ckpt:
        state_dict = torch.load(osp.join(args.save, args.reuse_ckpt_weight+'.pth'))
        model_instance.load_state_dict(state_dict)

    # Calculate source distribution & target distribution
    if args.use_oracle_IW:
        dset_s = ImageList(open(source_file).readlines(), root_folder=args.root_folder)
        dset_t = ImageList(open(target_file).readlines(), root_folder=args.root_folder)
        source_label_distribution = np.zeros((class_num))
        target_label_distribution = np.zeros((class_num))
        for img in dset_s.imgs:
            source_label_distribution[img[1]] += 1
        for img in dset_t.imgs:
            target_label_distribution[img[1]] += 1
        
        # Set the classess with too little amount to 1%
        max_num_images_source = np.max(source_label_distribution)
        for i, num_images in enumerate(source_label_distribution):
            if num_images <= 0.01*max_num_images_source:
                source_label_distribution[i] = 0.01*max_num_images_source
        max_num_images_target = np.max(target_label_distribution)
        for i, num_images in enumerate(target_label_distribution):
            if num_images <= 0.01*max_num_images_target:
                target_label_distribution[i] = 0.01*max_num_images_target

        source_label_distribution /= np.sum(source_label_distribution)
        target_label_distribution /= np.sum(target_label_distribution)

        # True importance weight   
        true_weights = torch.tensor(target_label_distribution / source_label_distribution, \
                                    dtype=torch.float, requires_grad=False)[:, None].cuda()
        print("Source label distribution: {}".format(source_label_distribution))
        print("Target label distribution: {}".format(target_label_distribution))
        print("True weights : {}".format(true_weights[:, 0].cpu().numpy()))
    else:
        true_weights = None
        source_label_distribution = None
        target_label_distribution = None
        print("Source label distribution: {}".format(source_label_distribution))
        print("Target label distribution: {}".format(target_label_distribution))
        print("True weights : {}".format(true_weights))


    train_source_loader = load_images(source_file, batch_size=args.batch_size, resize_size=resize_size, crop_size=crop_size, \
                                      is_cen=is_cen, root_folder=args.root_folder)

    if use_ssda:
        train_target_loader = load_images(target_file, batch_size=args.batch_size-args.labeled_tgt_batch_size, \
                                                resize_size=resize_size, crop_size=crop_size, is_cen=is_cen, \
                                                root_folder=args.root_folder)
        train_labeled_target_loader = load_images(labeled_target_file, batch_size=args.labeled_tgt_batch_size, \
                                                resize_size=resize_size, crop_size=crop_size, is_cen=is_cen, \
                                                root_folder=args.root_folder)
    else:
        train_target_loader = load_images(target_file, batch_size=args.batch_size, resize_size=resize_size, \
                                          crop_size=crop_size, is_cen=is_cen, root_folder=args.root_folder)
        train_labeled_target_loader = None

    test_source_loader = load_images(source_test_file, batch_size=16,  resize_size=resize_size, crop_size=crop_size, \
                                    is_train=False, is_cen=is_cen, root_folder=args.root_folder)
    test_target_loader = load_images(target_test_file, batch_size=16,  resize_size=resize_size, crop_size=crop_size, \
                                    is_train=False, is_cen=is_cen, root_folder=args.root_folder)

        
    train(args, model_instance, train_source_loader, train_target_loader, test_source_loader, test_target_loader,
          max_iter=300000,eval_interval=eval_interval, num_classes=class_num, 
          use_ssda=use_ssda, train_labeled_target_loader=train_labeled_target_loader, \
          true_weights=true_weights, source_weight=source_label_distribution)

