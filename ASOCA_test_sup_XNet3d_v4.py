from torchvision import transforms, datasets
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
import argparse
import os
import shutil
import numpy as np 
from torch.backends import cudnn
import torchio as tio
from dataload.test_util5 import test_all_case
from config.dataset_config.dataset_cfg import dataset_cfg
from config.augmentation.online_aug import data_transform_3d
from models.getnetwork import get_network
from config.train_test_config.train_test_config import save_test_3d
from warnings import simplefilter
os.environ['CUDA_VISIBLE_DEVICES']='2'


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str,default='ASOCA_DTCWT', help='Name of Experiment')
    parser.add_argument('-snapshot_path',default='xnetv3_3d_add-l=0.0001-e=601-s=6-g=0.99-b=2-w=20-100')
    parser.add_argument('-p', '--path_model', default='best_result1_Dc_0.8753.pth')
    parser.add_argument('--divided', default='/home/ac/datb/wch/checkpoints/div/ASOCA/', help='train_labeled.list,val.list')
    parser.add_argument('--cross_val', type=int,default=1, help='5-fold cross validation or random split 7/1/2 for training/validation/testing')
    #=1为交叉验证，=0为随机7/1/2划分数据
    parser.add_argument('--fold', type=int,default=3, help='cross validation')
    #第几折 比如分为4折交叉fold就被设置为0-3
    parser.add_argument('--dataset_name', default='imagecas', help='LiTS, Atrial')
    parser.add_argument('--exp', type=str, default='imagecas/FullSup/xnet', help='experiment_name')
    parser.add_argument('--input1', default='img')
    parser.add_argument('--input2', default= 'L' )
    parser.add_argument('--input3', default='HA')
    parser.add_argument('--input4', default='HR')
    parser.add_argument('--threshold', default=0.5000)
    parser.add_argument('--result', default='result1', help='result1, result2')
    parser.add_argument('--patch_size', default=(128,128,128))
    parser.add_argument('--patch_overlap', default=(64,64,64))
    parser.add_argument('-b', '--batch_size', default=1, type=int)
    parser.add_argument('-n', '--network', default='xnetv3_3d_add')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--labeled_ratio', type=int, default=1, help='1/labeled_ratio data is provided mask')
    # 切片中的步长为labeled_ratio意味着每隔labeled_ratio个元素选取一个元素,labeled_ratio=1:新列表包含了原列表中的所有元素

    args = parser.parse_args()
    root_path="/home/ac/datb/wch/"+args.root_path
    snapshot_path="/home/ac/datb/wch/checkpoints/sup/"+args.root_path+"/"+args.snapshot_path+"/fold"+str(args.fold)+"/"
    test_save_path=snapshot_path+"prediction/"
    path_model=snapshot_path+args.path_model
    if not os.path.exists(test_save_path):
        os.makedirs(test_save_path)
        
    # Config
    dataset_name = args.dataset_name
    cfg = dataset_cfg(dataset_name)
    num_classes=cfg['NUM_CLASSES']

    print_num = 42 + (cfg['NUM_CLASSES'] - 3) * 7
    print_num_minus = print_num - 2

    # Model
    model = get_network(args.network, cfg['IN_CHANNELS'], cfg['NUM_CLASSES'])
    model = model.cuda()
  
    print(os.path.exists(path_model))# 检查文件是否存在


    state_dict = torch.load(path_model,weights_only=True)# 加载预训练权重参数
    model.load_state_dict(state_dict=state_dict)
    print("init weight from {}".format(path_model))
    model.eval()
    if args.cross_val:
        val='val'+str(args.fold)+'.list'
    else:
        val='val.list'
        
    # print(model.lh_exchange_weight_L.data.clone().cpu().numpy())
    # print(model.lh_exchange_weight_H.data.clone().cpu().numpy())
    val_path = os.path.join(args.divided, val)
    avg_metric= test_all_case(model,root_path,method=args.network, test_list=val_path, num_classes=num_classes,
                                           patch_size=args.patch_size, stride_xy=64, stride_z=64,test_save_path=test_save_path)
    
    print(avg_metric)
