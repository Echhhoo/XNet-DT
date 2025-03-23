# nohup python ASOCA_train_sup_transbts.py >sup_transbts_fold00.out & 
from torchvision import transforms, datasets
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch.utils.data import DataLoader
import argparse
import time
import os
import numpy as np
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.backends import cudnn
import random
import torchio as tio
from config.dataset_config.dataset_cfg import dataset_cfg
from config.train_test_config.train_test_config import print_train_loss_sup, print_val_loss_sup, print_train_eval_sup, print_val_eval_sup, save_val_best_sup_3d, draw_pred_sup, print_best_sup
from config.visdom_config.visual_visdom import visdom_initialization_XNet, visualization_XNet
from config.warmup_config.warmup import GradualWarmupScheduler
from config.augmentation.online_aug import data_transform_3d
from loss.loss_function import segmentation_loss
from dataload.Dataset2 import BaseDataSets, RandomRotFlip, RandomCrop, ToTensor
from loss import losses
from models.getnetwork import get_network
from dataload.dataset_3d import dataset_iit
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)
from torch.cuda.amp import autocast, GradScaler
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True
import matplotlib.pyplot as plt


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_trained_models', default='/home/ac/datb/wch/checkpoints/sup')
    parser.add_argument('--path_seg_results', default='./seg_pred/sup_xnet')
    parser.add_argument('--path_dataset', default='/home/ac/datb/wch/ASOCA_h5')
    parser.add_argument('--root_path', type=str,default='/home/ac/datb/wch/ASOCA_h5', help='Name of Experiment')
    parser.add_argument('--dataset_name', default='imagecas', help='LiTS, Atrial')
    parser.add_argument('--input1', default='image')
    parser.add_argument('--sup_mark', default='100')
    parser.add_argument('-snapshot_path',default='/home/ac/datb/wch/checkpoints/div/ASOCA/')
    parser.add_argument('-b', '--batch_size', default=2, type=int)
    parser.add_argument('-e', '--num_epochs', default=600, type=int)
    parser.add_argument('-s', '--step_size', default=6, type=int)
    parser.add_argument('-l', '--lr', default=0.0001, type=float)
    parser.add_argument('-g', '--gamma', default=0.99, type=float)
    parser.add_argument('--fold', type=int,default=3, help='cross validation')#第几折 比如分为4折交叉fold就被设置为0-3
    parser.add_argument('--cross_val', type=int,default=1, help='5-fold cross validation or random split 7/1/2 for training/validation/testing')
    #=1为交叉验证，=0为随机7/1/2划分数据
    parser.add_argument('--optim', type=str, default='Adam',help='Optimizer types: Adam / AdamW / SGD')
    parser.add_argument('--patch_size', default=(128,128,128))
    parser.add_argument('--labeled_ratio', type=int, default=1, help='1/labeled_ratio data is provided mask')
    # 切片中的步长为labeled_ratio意味着每隔labeled_ratio个元素选取一个元素,labeled_ratio=1:新列表包含了原列表中的所有元素
    parser.add_argument('--loss', default='dice', type=str)
    parser.add_argument('-w', '--warm_up_duration', default=20)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--wd', default=-5, type=float, help='weight decay pow')
    parser.add_argument('--queue_length', default=48, type=int)
    parser.add_argument('--samples_per_volume_train', default=4, type=int)
    parser.add_argument('--samples_per_volume_val', default=8, type=int)

    parser.add_argument('-i', '--display_iter', default=1, type=int)
    parser.add_argument('-n', '--network', default='transbts', type=str)
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('-v', '--vis', default=True, help='need visualization or not')
    parser.add_argument('--visdom_port', default=16672, help='16672')
    args = parser.parse_args()

    dataset_name = args.dataset_name
    cfg = dataset_cfg(dataset_name)
    num_classes=cfg['NUM_CLASSES']

    print_num = 42 + (cfg['NUM_CLASSES'] - 3) * 7
    print_num_minus = print_num - 2

    path_trained_models = args.path_trained_models+'/'+str(os.path.split(args.path_dataset)[1]) 
    if not os.path.exists(path_trained_models):
        os.mkdir(path_trained_models)
    path_trained_models = path_trained_models+'/'+str(args.network)+'-l='+str(args.lr)+'-e='+str(args.num_epochs)+'-s='+str(args.step_size)+'-g='+str(args.gamma)+'-b='+str(args.batch_size)+'-w='+str(args.warm_up_duration)+'-'+str(args.sup_mark)
    if not os.path.exists(path_trained_models):
            os.mkdir(path_trained_models)
    path_trained_models = path_trained_models+'/fold'+str(args.fold)
    if not os.path.exists(path_trained_models):
            os.mkdir(path_trained_models)

    path_seg_results = path_trained_models+'/prediction'
    if not os.path.exists(path_seg_results):
            os.mkdir(path_seg_results)
        
    path_mask_results =path_trained_models + '/mask'
    if not os.path.exists(path_mask_results):
            os.mkdir(path_mask_results)
        

    dataset_train =BaseDataSets(base_dir=args.root_path, save_dir=args.snapshot_path, labeled_type="labeled",
                                labeled_ratio=args.labeled_ratio, fold=args.fold, split="train", 
                                transform=transforms.Compose([RandomCrop(args.patch_size),ToTensor(),]))
    dataset_val= BaseDataSets(base_dir=args.root_path, save_dir=args.snapshot_path, fold=args.fold,
                          split="val", labeled_ratio=args.labeled_ratio, cross_val=args.cross_val,
                          transform=transforms.Compose([RandomCrop(args.patch_size),ToTensor(),]))

    dataloaders = dict()
    dataloaders['train'] = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=8)
    dataloaders['val'] = DataLoader(dataset_val, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=8)

    num_batches = {'train_sup': len(dataloaders['train']), 'val': len(dataloaders['val'])}

    # Model
    model = get_network(args.network, cfg['IN_CHANNELS'], cfg['NUM_CLASSES'], img_shape=args.patch_size)
    model = model.cuda()


    # Training Strategy
    # criterion = segmentation_loss(args.loss, False).cuda()
    dice_loss = losses.DiceLoss(num_classes).cuda()

    if args.optim == 'AdamW':
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0001)
    elif args.optim == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    elif args.optim == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=5*10**args.wd)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    scheduler_warmup = GradualWarmupScheduler(optimizer, multiplier=1.0, total_epoch=args.warm_up_duration, after_scheduler=exp_lr_scheduler)


    # Train & Val
    since = time.time()
    count_iter = 0
    best_model = model
    best_result = 'Result1'
    best_val_eval_list = [0 for i in range(4)]
    dice_loss = losses.DiceLoss(num_classes).cuda()
    train_loss_total_list = []
    val_loss_list = []
    best_performance = 0.0

    for epoch in range(args.num_epochs):
        scaler =torch.amp.GradScaler('cuda')
        count_iter += 1
        if (count_iter-1) % args.display_iter == 0:
            begin_time = time.time()
        model.train()
        train_loss = 0.0
        train_loss = 0.0
        val_loss = 0.0

        for i,sampled_batch_labeled in enumerate(dataloaders['train']):

            inputs_train = Variable(sampled_batch_labeled['image'].cuda())
            mask_train = Variable(sampled_batch_labeled['label'].cuda())

            optimizer.zero_grad()
            with torch.amp.autocast('cuda',enabled=True):
                outputs_train = model(inputs_train)

                torch.cuda.empty_cache()

                if i == 0:
                    score_list_train = outputs_train
                    mask_list_train = mask_train
                else:
                # elif 0 < i <= num_batches['train_sup'] / 32:
                    score_list_train = torch.cat((score_list_train, outputs_train), dim=0)
                    mask_list_train = torch.cat((mask_list_train, mask_train), dim=0)
                loss_train=dice_loss(outputs_train,mask_train[:].unsqueeze(1))
            # loss_train = criterion(outputs_train, mask_train)
            scaler.scale(loss_train).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss_train.item()
        train_loss_total_list.append(train_loss/num_batches['train_sup'])
        scheduler_warmup.step()
        torch.cuda.empty_cache()

        if count_iter % args.display_iter == 0:
            torch.cuda.empty_cache()
            print('=' * print_num)
            print('| Epoch {}/{}'.format(epoch + 1, args.num_epochs).ljust(print_num_minus, ' '), '|')
            train_epoch_loss = print_train_loss_sup(train_loss, num_batches, print_num, print_num_minus)
            train_eval_list, train_m_jc = print_train_eval_sup(cfg['NUM_CLASSES'], score_list_train, mask_list_train, print_num_minus)
            torch.cuda.empty_cache()

            with torch.no_grad():
                model.eval()

                for i, sampled_batch_labeled in enumerate(dataloaders['val']):

                    # if 0 <= i <= num_batches['val']:
                    inputs_val = Variable(sampled_batch_labeled['image'].cuda())
                    mask_val = Variable(sampled_batch_labeled['label'].cuda())
                    
                    optimizer.zero_grad()
                    outputs_val = model(inputs_val)

                    torch.cuda.empty_cache()
                    if i == 0:
                        score_list_val = outputs_val
                        mask_list_val = mask_val
                    else:
                        score_list_val = torch.cat((score_list_val, outputs_val), dim=0)
                        mask_list_val = torch.cat((mask_list_val, mask_val), dim=0)
                        
                    loss_val=dice_loss(outputs_val,mask_val[:].unsqueeze(1))


                    # loss_val = criterion(outputs_val, mask_val)
                    val_loss += loss_val.item()
                val_loss_list.append(val_loss/num_batches['val'])
                torch.cuda.empty_cache()

                val_epoch_loss = print_val_loss_sup(val_loss, num_batches, print_num, print_num_minus)
                val_eval_list, val_m_jc = print_val_eval_sup(cfg['NUM_CLASSES'], score_list_val, mask_list_val, print_num_minus)
                best_val_eval_list = save_val_best_sup_3d(cfg['NUM_CLASSES'], best_val_eval_list, model, score_list_val, mask_list_val, val_eval_list, path_trained_models, path_seg_results, path_mask_results, args.network, cfg['FORMAT'])
                torch.cuda.empty_cache()
                print('-' * print_num)
                print('| Epoch Time: {:.4f}s'.format((time.time() - begin_time) / args.display_iter).ljust(print_num_minus, ' '), '|')
            torch.cuda.empty_cache()
        torch.cuda.empty_cache()
    epochs = range(1, args.num_epochs + 1,args.display_iter)
    plt.plot(epochs, train_loss_total_list, label='train_loss')
    plt.plot(epochs, val_loss_list, label='val_loss')

    plt.title('Training Losses over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('sup_transbts_fold03.png')

    plt.show()
    # if rank == args.rank_index:
    #     time_elapsed = time.time() - since
    #     m, s = divmod(time_elapsed, 60)
    #     h, m = divmod(m, 60)

    #     print('=' * print_num)
    #     print('| Training Completed In {:.0f}h {:.0f}mins {:.0f}s'.format(h, m, s).ljust(print_num_minus, ' '), '|')
    #     print('-' * print_num)
    #     print_best_sup(cfg['NUM_CLASSES'], best_val_eval_list, print_num_minus)
    #     print('=' * print_num)