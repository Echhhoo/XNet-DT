# nohup python ASOCA_train_semi_URPC.py >semi_ASOCA_urpc_fold00.out &
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
from torch.backends import cudnn
import random
import torchio as tio

from config.dataset_config.dataset_cfg import dataset_cfg
from config.train_test_config.train_test_config import print_train_loss_EM, print_val_loss_sup, print_train_eval_sup, print_val_eval_sup, save_val_best_sup_3d, print_best_sup
from config.visdom_config.visual_visdom import visdom_initialization_EM, visualization_EM
from config.warmup_config.warmup import GradualWarmupScheduler
from dataload.Dataset2 import BaseDataSets, RandomCrop, ToTensor
from loss import losses
from models.getnetwork import get_network
from dataload.dataset_3d import dataset_iit
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)
from torch.cuda.amp import autocast, GradScaler
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True
import matplotlib.pyplot as plt
from config.ramps import ramps


def count_parameters(model):
    param_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    KB = 1024
    MB = 1024 * KB
    GB = 1024 * MB
    if param_num / GB >= 1:
        return f"{param_num / GB:.2f} GB"
    elif param_num / MB >= 1:
        return f"{param_num / MB:.2f} MB"
    elif param_num / KB >= 1:
        return f"{param_num / KB:.2f} KB"
    else:
        return f"{param_num} 参数个数"


simplefilter(action='ignore', category=FutureWarning)


def init_seeds(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_trained_models', default='/home/ac/datb/wch/checkpoints/semi')
    parser.add_argument('--path_seg_results', default='./seg_pred/sup_xnet')
    parser.add_argument('--path_dataset', default='/home/ac/datb/wch/ASOCA_h5')
    parser.add_argument('--root_path', type=str,default='/home/ac/datb/wch/ASOCA_h5', help='Name of Experiment')
    parser.add_argument('-snapshot_path',default='/home/ac/datb/wch/checkpoints/sup/ASOCA_DTCWT/')
    parser.add_argument('--dataset_name', default='imagecas', help='LiTS, Atrial')
    parser.add_argument('--input1', default='image')
    parser.add_argument('--sup_mark', default='20')
    parser.add_argument('--unsup_mark', default='80')
    parser.add_argument('-b', '--batch_size', default=2, type=int)
    parser.add_argument('-e', '--num_epochs', default=600, type=int)
    parser.add_argument('-s', '--step_size', default=6, type=int)
    parser.add_argument('-l', '--lr', default=0.0001, type=float)
    parser.add_argument('-g', '--gamma', default=0.99, type=float)
    parser.add_argument('-u', '--unsup_weight', default=1, type=float)
    parser.add_argument('--fold', type=int,default=2, help='cross validation')#第几折 比如分为4折交叉fold就被设置为0-3
    parser.add_argument('--cross_val', type=int,default=1, help='5-fold cross validation or random split 7/1/2 for training/validation/testing')
    #=1为交叉验证，=0为随机7/1/2划分数据
    parser.add_argument('--optim', type=str, default='Adam',help='Optimizer types: Adam / AdamW / SGD')
    parser.add_argument('--patch_size', default=(128,128,128))
    parser.add_argument('--labeled_ratio', type=int, default=3, help='1/labeled_ratio data is provided mask')
    # 切片中的步长为labeled_ratio意味着每隔labeled_ratio个元素选取一个元素,labeled_ratio=1:新列表包含了原列表中的所有元素
    parser.add_argument('--loss', default='dice')
    parser.add_argument('-w', '--warm_up_duration', default=20)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--ema_decay', default=0.99, type=float)
    parser.add_argument('--wd', default=-5, type=float, help='weight decay pow')

    parser.add_argument('-i', '--display_iter', default=1, type=int)
    parser.add_argument('-n', '--network', default='unet_urpc', type=str)
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--rank_index', default=0, help='0, 1, 2, 3')
    parser.add_argument('-v', '--vis', default=True, help='need visualization or not')
    args = parser.parse_args()

    dataset_name = args.dataset_name
    cfg = dataset_cfg(dataset_name)
    num_classes=cfg['NUM_CLASSES']

    print_num = 77 + (cfg['NUM_CLASSES'] - 3) * 14
    print_num_minus = print_num - 2
    print_num_half = int(print_num / 2 - 1)

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

    # Dataset
   
    dataset_train_unsup =BaseDataSets(base_dir=args.root_path, save_dir=args.snapshot_path, labeled_type="unlabeled",
                                labeled_ratio=args.labeled_ratio, fold=args.fold, split="train", 
                                transform=transforms.Compose([RandomCrop(args.patch_size),ToTensor(),]))
    dataset_train_sup =BaseDataSets(base_dir=args.root_path, save_dir=args.snapshot_path, labeled_type="labeled",
                                labeled_ratio=args.labeled_ratio, fold=args.fold, split="train", 
                                transform=transforms.Compose([RandomCrop(args.patch_size),ToTensor(),]))
    dataset_val= BaseDataSets(base_dir=args.root_path, save_dir=args.snapshot_path, fold=args.fold,
                          split="val", labeled_ratio=args.labeled_ratio, cross_val=args.cross_val,
                          transform=transforms.Compose([RandomCrop(args.patch_size),ToTensor(),]))

    dataloaders = dict()
    dataloaders['train_sup'] = DataLoader(dataset_train_sup, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=8)
    dataloaders['train_unsup'] = DataLoader(dataset_train_unsup, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=8)
    dataloaders['val'] = DataLoader(dataset_val, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=8)

    num_batches = {'train_sup': len(dataloaders['train_sup']), 'train_unsup': len(dataloaders['train_unsup']), 'val': len(dataloaders['val'])}

    # Model
    model1 = get_network(args.network, cfg['IN_CHANNELS'], cfg['NUM_CLASSES'])

    model1 = model1.cuda()
    param_num = count_parameters(model1)
    print(f"模型的参数量为: {param_num}")
    

    # Training Strategy
    if args.optim == 'AdamW':
        optimizer1 = torch.optim.AdamW(model1.parameters(), lr=args.lr, weight_decay=0.0001)
    elif args.optim == 'Adam':
        optimizer1 = torch.optim.Adam(model1.parameters(), lr=args.lr)
    elif args.optim == 'SGD':
        optimizer1 = optim.SGD(model1.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=5*10**args.wd)
    exp_lr_scheduler1 = lr_scheduler.StepLR(optimizer1, step_size=args.step_size, gamma=args.gamma)
    scheduler_warmup1 = GradualWarmupScheduler(optimizer1, multiplier=1.0, total_epoch=args.warm_up_duration, after_scheduler=exp_lr_scheduler1)

    dice_loss = losses.DiceLoss(num_classes).cuda()
    kl_distance = nn.KLDivLoss(reduction='none')
    
    since = time.time()
    count_iter = 0

    best_model = model1
    best_result = 'Result1'
    best_val_eval_list = [0 for i in range(4)]
    train_loss_total_list = []
    val_loss_list = []
    best_performance = 0.0

   

    for epoch in range(args.num_epochs):

        count_iter += 1
        if (count_iter - 1) % args.display_iter == 0:
            begin_time = time.time()

        model1.train()

        train_loss_sup_1 = 0.0
        train_loss_unsup = 0.0
        train_loss = 0.0

        val_loss_sup_1 = 0.0

        unsup_weight = args.unsup_weight * (epoch + 1) / args.num_epochs


        dataset_train_sup = iter(dataloaders['train_sup'])
        dataset_train_unsup = iter(dataloaders['train_unsup'])

        for i in range(num_batches['train_sup']):

            unsup_index = next(dataset_train_unsup)
            img_train_unsup_1 = Variable(unsup_index['image'].cuda(non_blocking=True))

            optimizer1.zero_grad()

            pred_train_unsup1, pred_train_unsup2, pred_train_unsup3, pred_train_unsup4 = model1(img_train_unsup_1)
            pred_train_unsup1 = torch.softmax(pred_train_unsup1, 1)
            pred_train_unsup2 = torch.softmax(pred_train_unsup2, 1)
            pred_train_unsup3 = torch.softmax(pred_train_unsup3, 1)
            pred_train_unsup4 = torch.softmax(pred_train_unsup4, 1)

            preds = (pred_train_unsup1 + pred_train_unsup2 + pred_train_unsup3 + pred_train_unsup4) / 4

            variance_aux1 = torch.sum(kl_distance(torch.log(pred_train_unsup1 + 1e-6), preds), dim=1, keepdim=True)
            exp_variance_aux1 = torch.exp(-variance_aux1)

            variance_aux2 = torch.sum(kl_distance(torch.log(pred_train_unsup2 + 1e-6), preds), dim=1, keepdim=True)
            exp_variance_aux2 = torch.exp(-variance_aux2)

            variance_aux3 = torch.sum(kl_distance(torch.log(pred_train_unsup3 + 1e-6), preds), dim=1, keepdim=True)
            exp_variance_aux3 = torch.exp(-variance_aux3)

            variance_aux4 = torch.sum(kl_distance(torch.log(pred_train_unsup4 + 1e-6), preds), dim=1, keepdim=True)
            exp_variance_aux4 = torch.exp(-variance_aux4)

            consistency_dist_aux1 = (preds - pred_train_unsup1) ** 2
            consistency_loss_aux1 = torch.mean(consistency_dist_aux1 * exp_variance_aux1) / (torch.mean(exp_variance_aux1) + 1e-8) + torch.mean(variance_aux1)

            consistency_dist_aux2 = (preds - pred_train_unsup2) ** 2
            consistency_loss_aux2 = torch.mean(consistency_dist_aux2 * exp_variance_aux2) / (torch.mean(exp_variance_aux2) + 1e-8) + torch.mean(variance_aux2)

            consistency_dist_aux3 = (preds - pred_train_unsup3) ** 2
            consistency_loss_aux3 = torch.mean(consistency_dist_aux3 * exp_variance_aux3) / (torch.mean(exp_variance_aux3) + 1e-8) + torch.mean(variance_aux3)

            consistency_dist_aux4 = (preds - pred_train_unsup4) ** 2
            consistency_loss_aux4 = torch.mean(consistency_dist_aux4 * exp_variance_aux4) / (torch.mean(exp_variance_aux4) + 1e-8) + torch.mean(variance_aux4)

            loss_train_unsup = (consistency_loss_aux1 + consistency_loss_aux2 + consistency_loss_aux3 + consistency_loss_aux4) / 4

            loss_train_unsup = loss_train_unsup * unsup_weight
            loss_train_unsup.backward(retain_graph=True)
            torch.cuda.empty_cache()

            sup_index = next(dataset_train_sup)
            img_train_sup_1 = Variable(sup_index['image'].cuda(non_blocking=True))
            mask_train_sup = Variable(sup_index['label'].cuda(non_blocking=True))

            pred_train_sup1, pred_train_sup2, pred_train_sup3, pred_train_sup4 = model1(img_train_sup_1)

            if count_iter % args.display_iter == 0:
                if i == 0:
                    score_list_train1 = pred_train_sup1
                    mask_list_train = mask_train_sup
                # else:
                elif 0 < i <= num_batches['train_sup'] / 32:
                    score_list_train1 = torch.cat((score_list_train1, pred_train_sup1), dim=0)
                    mask_list_train = torch.cat((mask_list_train, mask_train_sup), dim=0)

            loss_train_sup1 = (dice_loss(pred_train_sup1, mask_train_sup[:].unsqueeze(1))+dice_loss(pred_train_sup2, mask_train_sup[:].unsqueeze(1))+dice_loss(pred_train_sup3, mask_train_sup[:].unsqueeze(1))+dice_loss(pred_train_sup4, mask_train_sup[:].unsqueeze(1))) / 4
            loss_train_sup = loss_train_sup1

            loss_train_sup.backward()
            optimizer1.step()
            torch.cuda.empty_cache()

            loss_train = loss_train_unsup + loss_train_sup
            train_loss_unsup += loss_train_unsup.item()
            train_loss_sup_1 += loss_train_sup1.item()
            train_loss += loss_train.item()

        scheduler_warmup1.step()
        torch.cuda.empty_cache()

        if count_iter % args.display_iter == 0:
            torch.cuda.empty_cache()
            print('=' * print_num)
            print('| Epoch {}/{}'.format(epoch + 1, args.num_epochs).ljust(print_num_minus, ' '), '|')
            train_epoch_loss_sup_1, train_epoch_loss_cps, train_epoch_loss = print_train_loss_EM(train_loss_sup_1, train_loss_unsup, train_loss, num_batches, print_num, print_num_minus)
            train_eval_list_1, train_m_jc_1 = print_train_eval_sup(cfg['NUM_CLASSES'], score_list_train1, mask_list_train, print_num_minus)
            torch.cuda.empty_cache()

            with torch.no_grad():
                model1.eval()

                for i, sampled_batch_labeled in enumerate(dataloaders['val']):


                    # if 0 <= i <= num_batches['val'] / 16:
                    inputs_val_1= Variable(sampled_batch_labeled['image'].cuda())
                    mask_val = Variable(sampled_batch_labeled['label'].cuda())

                    optimizer1.zero_grad()
                    outputs_val_1, outputs_val_2, outputs_val_3, outputs_val_4 = model1(inputs_val_1)
                    torch.cuda.empty_cache()

                    if i == 0:
                        score_list_val_1 = outputs_val_1
                        mask_list_val = mask_val
                    else:
                        score_list_val_1 = torch.cat((score_list_val_1, outputs_val_1), dim=0)
                        mask_list_val = torch.cat((mask_list_val, mask_val), dim=0)

                    loss_val_sup_1 = dice_loss(outputs_val_1, mask_val[:].unsqueeze(1))
                    val_loss_sup_1 += loss_val_sup_1.item()

                torch.cuda.empty_cache()
                

                
                val_epoch_loss_sup_1 = print_val_loss_sup(val_loss_sup_1, num_batches, print_num, print_num_minus)
                val_eval_list_1, val_m_jc_1 = print_val_eval_sup(cfg['NUM_CLASSES'], score_list_val_1, mask_list_val, print_num_minus)
                best_val_eval_list = save_val_best_sup_3d(cfg['NUM_CLASSES'], best_val_eval_list, model1, score_list_val_1, mask_list_val, val_eval_list_1, path_trained_models, path_seg_results, path_mask_results, 'URPC', cfg['FORMAT'])
                torch.cuda.empty_cache()
                print('-' * print_num)
                print('| Epoch Time: {:.4f}s'.format((time.time() - begin_time) / args.display_iter).ljust(print_num_minus, ' '), '|')
            torch.cuda.empty_cache()
        torch.cuda.empty_cache()

    # if rank == args.rank_index:
    #     time_elapsed = time.time() - since
    #     m, s = divmod(time_elapsed, 60)
    #     h, m = divmod(m, 60)

    #     print('=' * print_num)
    #     print('| Training Completed In {:.0f}h {:.0f}mins {:.0f}s'.format(h, m, s).ljust(print_num_minus, ' '), '|')
    #     print('-' * print_num)
    #     print_best_sup(cfg['NUM_CLASSES'], best_val_eval_list, print_num_minus)
    #     print('=' * print_num)