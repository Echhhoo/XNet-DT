# nohup python ASOCA_train_sup_v2_dice.py >sup_xnetv2_dwt_dice_fold00.out &
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
from torch.nn.modules.loss import CrossEntropyLoss
from torch.backends import cudnn
import random
import torchio as tio
import pywt
from torchio import transforms as T
from loss.cbdice_loss import SoftcbDiceLoss
from config.dataset_config.dataset_cfg import dataset_cfg
from config.train_test_config.train_xnetv2_config import print_train_loss_XNetv2, print_val_loss_XNetv2, print_train_eval_sup, print_val_eval_sup, save_val_best_sup_3d, print_best_sup
from config.warmup_config.warmup import GradualWarmupScheduler
from config.augmentation.online_aug import data_transform_3d
from loss.loss_function import segmentation_loss
from models.getnetwork import get_network
from dataload.dataset_3d import dataset_it
from dataload.Dataset4 import BaseDataSets, RandomRotFlip, RandomCrop, ToTensor
from loss import losses
from models.getnetwork import get_network
from warnings import simplefilter
from torch.cuda.amp import autocast, GradScaler
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True
import matplotlib.pyplot as plt




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_trained_models', default='/home/ac/datb/wch/checkpoints/sup')
    parser.add_argument('--path_seg_results', default='./seg_pred/sup_xnet')
    parser.add_argument('--path_dataset', default='/home/ac/datb/wch/ASOCA_DWT')
    parser.add_argument('--root_path', type=str,default='/home/ac/datb/wch/ASOCA_DWT', help='Name of Experiment')
    parser.add_argument('--dataset_name', default='imagecas', help='LiTS, Atrial')
    parser.add_argument('-snapshot_path',default='/home/ac/datb/wch/checkpoints/sup/ASOCA_DTCWT/')
    parser.add_argument('--input1', default='img')
    parser.add_argument('--input2', default='L')
    parser.add_argument('--input3', default='H')
    parser.add_argument('--sup_mark', default='100', help='100')
    parser.add_argument('--exp', type=str, default='imagecas/FullSup/xnet', help='experiment_name')
    parser.add_argument('--fold', type=int,default=3, help='cross validation')#第几折 比如分为4折交叉fold就被设置为0-3
    parser.add_argument('--cross_val', type=int,default=1, help='4-fold cross validation or random split 7/1/2 for training/validation/testing')
    #=1为交叉验证，=0为随机7/1/2划分数据
    parser.add_argument('--optim', type=str, default='Adam',help='Optimizer types: Adam / AdamW / SGD')
    parser.add_argument('-b', '--batch_size', default=2, type=int)
    parser.add_argument('-e', '--num_epochs', default=600, type=int)
    parser.add_argument('-s', '--step_size', default=6, type=int)
    parser.add_argument('-l', '--lr', default=0.0001, type=float)
    parser.add_argument('-g', '--gamma', default=0.99, type=float)
    parser.add_argument('-u', '--unsup_weight', default=0.5, type=float)
    parser.add_argument('--loss', default='dice', type=str)
    parser.add_argument('--patch_size', default=(128,128,128))
    parser.add_argument('--labeled_ratio', type=int, default=1, help='1/labeled_ratio data is provided mask')
    # 切片中的步长为labeled_ratio意味着每隔labeled_ratio个元素选取一个元素,labeled_ratio=1:新列表包含了原列表中的所有元素
    parser.add_argument('-w', '--warm_up_duration', default=20)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--wd', default=-5, type=float, help='weight decay pow')
    parser.add_argument('--queue_length', default=48, type=int)
    parser.add_argument('--samples_per_volume_train', default=4, type=int)
    parser.add_argument('--samples_per_volume_val', default=8, type=int)
    # parser.add_argument('--train_alpha', default=[0.0, 0.0])
    # parser.add_argument('--train_beta', default=[0.2, 0.2])
    # parser.add_argument('--val_alpha', default=[0.0, 0.0])
    # parser.add_argument('--val_beta', default=[0.2, 0.2])

    parser.add_argument('--display_iter', default=1, type=int)
    parser.add_argument('-n', '--network', default='xnetv2_3d_min', type=str)
    
    # parser.add_argument('--deterministic', type=int,  default=0,help='whether use deterministic training')
    # parser.add_argument('--seed', type=int,  default=42, help='random seed')
    args = parser.parse_args()
    
    
    # if not args.deterministic:
    #     cudnn.benchmark = True
    #     cudnn.deterministic = False
    # else:
    #     cudnn.benchmark = False
    #     cudnn.deterministic = True

    # random.seed(args.seed)
    # np.random.seed(args.seed)
    # torch.manual_seed(args.seed)
    # torch.cuda.manual_seed(args.seed)


    dataset_name = args.dataset_name
    cfg = dataset_cfg(dataset_name)
    num_classes=cfg['NUM_CLASSES']

    print_num = 77 + (cfg['NUM_CLASSES'] - 3) * 14
    print_num_minus = print_num - 2
    print_num_half = int(print_num / 3 - 1)

    path_trained_models = args.path_trained_models + '/' + str(os.path.split(args.path_dataset)[1])
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
    model = get_network(args.network, cfg['IN_CHANNELS'], cfg['NUM_CLASSES'])
    model = model.cuda()

     # Training Strategy
    # criterion = segmentation_loss(args.loss, False).cuda()

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
    ce_loss = CrossEntropyLoss()
    cbdice_loss=SoftcbDiceLoss()
    
    train_loss_total_list = []
    val_loss_sup1_list=[]  
    val_loss_sup2_list=[]
    val_loss_sup3_list=[]
    L_exchange=[]
    H_exchange=[]
    best_performance = 0.0


    for epoch in range(args.num_epochs):
        scaler =torch.amp.GradScaler('cuda')
        count_iter += 1
        if (count_iter-1) % args.display_iter == 0:
            begin_time = time.time()
        model.train()

        train_loss_sup_1 = 0.0
        train_loss_sup_2 = 0.0
        train_loss_sup_3 = 0.0
        train_loss_unsup = 0.0
        train_loss = 0.0

        val_loss_sup_1 = 0.0
        val_loss_sup_2 = 0.0
        val_loss_sup_3 = 0.0
        unsup_weight = args.unsup_weight * (epoch + 1) / args.num_epochs
        # print(model.lh_exchange_weight_L.data.clone().cpu().numpy())
        # print(model.lh_exchange_weight_H.data.clone().cpu().numpy())
            

        for i, sampled_batch_labeled in enumerate(dataloaders['train']):
            inputs_train_1 = Variable(sampled_batch_labeled['image1'].cuda())
            inputs_train_2 = Variable(sampled_batch_labeled['image2'].cuda())
            inputs_train_3 = Variable(sampled_batch_labeled['image3'].cuda())
            mask_train = Variable(sampled_batch_labeled['label'].cuda())
            
            # L_alpha = random.uniform(args.train_alpha[0], args.train_alpha[1])
            # L = inputs_train_2 + L_alpha * inputs_train_3
            # L = (L -torch.mean(L)) / torch.std(L)

            # H_beta = random.uniform(args.train_beta[0], args.train_beta[1])
            # H = inputs_train_2 + H_beta * inputs_train_3
            # H = (H -torch.mean(H)) / torch.std(H)
            # inputs_train_2 = Variable(L.cuda())
            # inputs_train_3 = Variable(H.cuda())
            

            optimizer.zero_grad()
            with torch.amp.autocast('cuda',enabled=True):  
                outputs_train_1, outputs_train_2 ,outputs_train_3=  model(inputs_train_1, inputs_train_2, inputs_train_3)
                torch.cuda.empty_cache()

                if i == 0:
                    score_list_train_1 = outputs_train_1
                    score_list_train_2 = outputs_train_2
                    score_list_train_3 = outputs_train_3
                    mask_list_train = mask_train
                else:
                # elif 0 < i <= num_batches['train_sup'] / 64:
                    score_list_train_1 = torch.cat((score_list_train_1, outputs_train_1), dim=0)
                    score_list_train_2 = torch.cat((score_list_train_2, outputs_train_2), dim=0)
                    score_list_train_3 = torch.cat((score_list_train_3, outputs_train_3), dim=0)
                    mask_list_train = torch.cat((mask_list_train, mask_train), dim=0)

       
                max_train1 = torch.max(outputs_train_1, dim=1)[1].long()
                max_train2 = torch.max(outputs_train_2, dim=1)[1].long()
                max_train3 = torch.max(outputs_train_3, dim=1)[1].long()
            # L_exchange.append(model.lh_exchange_weight_L.data.clone().cpu().numpy()) 
            # H_exchange.append(model.lh_exchange_weight_H.data.clone().cpu().numpy()) 
                loss_train_sup1_dice = dice_loss(outputs_train_1, mask_train[:].unsqueeze(1))
            # loss_train_sup1_ce=ce_loss(outputs_train_1,mask_train[:].long())
            # loss_train_sup1_cbdice=cbdice_loss(outputs_train_1, mask_train[:].unsqueeze(1))
            # loss_train_sup1=0.5*loss_train_sup1_dice+0.5*loss_train_sup1_ce
                loss_train_sup1=loss_train_sup1_dice
                loss_train_sup2_dice = dice_loss(outputs_train_2, mask_train[:].unsqueeze(1))
            # loss_train_sup2_ce=ce_loss(outputs_train_2,mask_train[:].long())
            # loss_train_sup2_cbdice=cbdice_loss(outputs_train_2, mask_train[:].unsqueeze(1))
            # loss_train_sup2=0.5*loss_train_sup2_dice+0.5*loss_train_sup2_ce
                loss_train_sup2=loss_train_sup2_dice
                loss_train_sup3_dice = dice_loss(outputs_train_3, mask_train[:].unsqueeze(1))
            # loss_train_sup3_ce=ce_loss(outputs_train_3,mask_train[:].long())
            # loss_train_sup3_cbdice=cbdice_loss(outputs_train_3, mask_train[:].unsqueeze(1))
            # loss_train_sup3=0.5*loss_train_sup3_dice+0.5*loss_train_sup3_ce
                loss_train_sup3=loss_train_sup3_dice
                loss_train_unsup_dice = dice_loss(outputs_train_1, max_train2[:].unsqueeze(1)) + dice_loss(outputs_train_2, max_train1[:].unsqueeze(1))+dice_loss(outputs_train_1, max_train3[:].unsqueeze(1)) + dice_loss(outputs_train_3, max_train1[:].unsqueeze(1))
            # loss_train_unsup_ce=ce_loss(outputs_train_1,max_train2[:].long())+ce_loss(outputs_train_2,max_train1[:].long())+ce_loss(outputs_train_1,max_train3[:].long())+ce_loss(outputs_train_3,max_train1[:].long())
            
            # loss_train_unsup=0.5*loss_train_unsup_dice+0.5*loss_train_unsup_ce
                loss_train_unsup=loss_train_unsup_dice
                loss_train_unsup = loss_train_unsup * unsup_weight
                loss_train = loss_train_sup1 + loss_train_sup2 + loss_train_sup3 + loss_train_unsup
            

            # loss_train.backward()
            # optimizer.step()
            scaler.scale(loss_train).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss_sup_1 += loss_train_sup1.item()
            train_loss_sup_2 += loss_train_sup2.item()
            train_loss_sup_3 += loss_train_sup3.item()
            train_loss_unsup += loss_train_unsup.item()
            train_loss += loss_train.item()

        scheduler_warmup.step()
        torch.cuda.empty_cache()
        patch_size=args.patch_size
        train_loss_total_list.append(train_loss/num_batches['train_sup'])
        if count_iter % args.display_iter == 0:
            torch.cuda.empty_cache()
            print('=' * print_num)
            print('| Epoch {}/{}'.format(epoch + 1, args.num_epochs).ljust(print_num_minus, ' '), '|')
            train_epoch_loss_sup_1, train_epoch_loss_sup_2, train_epoch_loss_sup_3, train_epoch_loss_unsup, train_epoch_loss = print_train_loss_XNetv2(train_loss_sup_1, train_loss_sup_2, train_loss_sup_3, train_loss_unsup, train_loss, num_batches, print_num, print_num_minus)
            train_eval_list_1, train_m_jc_1 = print_train_eval_sup(cfg['NUM_CLASSES'], score_list_train_1, mask_list_train, print_num_minus)
            torch.cuda.empty_cache()

            with torch.no_grad():
                model.eval()

                 
                for i, sampled_batch_labeled in enumerate(dataloaders['val']):

                    inputs_val_1 = Variable(sampled_batch_labeled['image1'].cuda())
                    inputs_val_2 = Variable(sampled_batch_labeled['image2'].cuda())
                    inputs_val_3 = Variable(sampled_batch_labeled['image3'].cuda())
                    mask_val = Variable(sampled_batch_labeled['label'].cuda())
                    
                    # L_alpha = random.uniform(args.val_alpha[0], args.val_alpha[1])
                    # L = inputs_val_2 + L_alpha * inputs_val_3
                    # L = (L -torch.mean(L)) / torch.std(L)

                    # H_beta = random.uniform(args.val_beta[0], args.val_beta[1])
                    # H = inputs_val_2 + H_beta * inputs_val_3
                    # H = (H -torch.mean(H)) / torch.std(H)
                    # inputs_val_2 = Variable(L.cuda())
                    # inputs_val_3 = Variable(H.cuda())
                    
                    optimizer.zero_grad()
                    outputs_val_1, outputs_val_2 ,outputs_val_3 = model(inputs_val_1, inputs_val_2, inputs_val_3)
                    torch.cuda.empty_cache()
                    
                    if i == 0:
                        score_list_val_1 = outputs_val_1
                        score_list_val_2 = outputs_val_2
                        score_list_val_3 = outputs_val_3
                        mask_list_val = mask_val
                    else:
                        score_list_val_1 = torch.cat((score_list_val_1, outputs_val_1), dim=0)
                        score_list_val_2 = torch.cat((score_list_val_2, outputs_val_2), dim=0)
                        score_list_val_3 = torch.cat((score_list_val_3, outputs_val_3), dim=0)
                        mask_list_val = torch.cat((mask_list_val, mask_val), dim=0)
                       
                    # loss_val_sup_1 = 0.5*dice_loss(outputs_val_1, mask_val[:].unsqueeze(1))+0.5*ce_loss(outputs_val_1,mask_val[:].long())
                    # loss_val_sup_2 = 0.5*dice_loss(outputs_val_2, mask_val[:].unsqueeze(1))+0.5*ce_loss(outputs_val_2,mask_val[:].long())
                    # loss_val_sup_3 = 0.5*dice_loss(outputs_val_3, mask_val[:].unsqueeze(1))+0.5*ce_loss(outputs_val_3,mask_val[:].long())
                    loss_val_sup_1 = dice_loss(outputs_val_1, mask_val[:].unsqueeze(1))
                    loss_val_sup_2 = dice_loss(outputs_val_2, mask_val[:].unsqueeze(1))
                    loss_val_sup_3 = dice_loss(outputs_val_3, mask_val[:].unsqueeze(1))
                    val_loss_sup_1 += loss_val_sup_1.item()
                    val_loss_sup_2 += loss_val_sup_2.item()
                    val_loss_sup_3 += loss_val_sup_3.item()

                val_loss_sup1_list.append(val_loss_sup_1 / num_batches['val'])  
                val_loss_sup2_list.append(val_loss_sup_2 / num_batches['val'])
                val_loss_sup3_list.append(val_loss_sup_3 / num_batches['val'])  
    
                torch.cuda.empty_cache()
                val_epoch_loss_sup_1, val_epoch_loss_sup_2, val_epoch_loss_sup_3 = print_val_loss_XNetv2(val_loss_sup_1, val_loss_sup_2, val_loss_sup_3, num_batches, print_num, print_num_half)
                val_eval_list_1, val_eval_list_2, val_eval_list_3, val_m_jc_1,val_m_jc_2,val_m_jc_3= print_val_eval_sup(cfg['NUM_CLASSES'], score_list_val_1,score_list_val_2,score_list_val_3, mask_list_val, print_num_half)
                best_val_eval_list, best_model, best_result = save_val_best_sup_3d(best_model, best_val_eval_list,best_result, model, model, model,val_eval_list_1, val_eval_list_2, val_eval_list_3,path_trained_models, 'XNetv2')
                torch.cuda.empty_cache()
                print('-' * print_num)
                print('| Epoch Time: {:.4f}s'.format((time.time() - begin_time) / args.display_iter).ljust(print_num_minus, ' '), '|')
            torch.cuda.empty_cache()
        torch.cuda.empty_cache()
    print("Training Finished!")
    epochs = range(1, args.num_epochs + 1, args.display_iter)

    plt.plot(epochs, train_loss_total_list, label='train_loss_total')
    plt.plot(epochs, val_loss_sup1_list, label='val_loss_sup_1')
    plt.plot(epochs, val_loss_sup2_list, label='val_loss_sup_2')
    plt.plot(epochs, val_loss_sup3_list, label='val_loss_sup_3')

    plt.title('Training Losses over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('sup_xnetv2_dwt_dice_fold03.png')
    
    # plt.plot(epochs, L_exchange, label='L_exchange')
    # plt.plot(epochs, H_exchange, label='H_exchange')
 

    # plt.title('lh_exchange')
    # plt.xlabel('Epochs')
    # plt.ylabel('lh_')
    # plt.legend()
    # plt.savefig('lh_change_fold_111302.png')

    