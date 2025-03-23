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
from dataload.test_util2 import test_all_case
from config.dataset_config.dataset_cfg import dataset_cfg
from config.augmentation.online_aug import data_transform_3d
from models.getnetwork import get_network
from models.networks_3d.uxnet import UXNET
from models.networks_3d.DSCNet import DSCNet
os.environ['CUDA_VISIBLE_DEVICES']='2'


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str,default='ASOCA_h5', help='Name of Experiment')
    parser.add_argument('-snapshot_path',default='unet3d64-l=0.0001-e=600-s=6-g=0.99-b=2-w=20-100')
    parser.add_argument('-p', '--path_model', default='best_unet3d_Dc_0.9038.pth')
    parser.add_argument('--divided', default='/home/ac/datb/wch/checkpoints/div/ASOCA/', help='train_labeled.list,val.list')
    parser.add_argument('--cross_val', type=int,default=1, help='5-fold cross validation or random split 7/1/2 for training/validation/testing')
    #=1为交叉验证，=0为随机7/1/2划分数据
    parser.add_argument('--fold', type=int,default=2, help='cross validation')#第几折 比如分为4折交叉fold就被设置为0-3
    parser.add_argument('--dataset_name', default='imagecas', help='LiTS, Atrial')
    parser.add_argument('--exp', type=str, default='imagecas/FullSup/xnet', help='experiment_name')
    parser.add_argument('--input1', default='H')
    parser.add_argument('--input2', default='L')
    parser.add_argument('--threshold', default=0.5)
    parser.add_argument('--result', default='result1', help='result1, result2')
    parser.add_argument('--patch_size', default=(128,128,128))
    parser.add_argument('--patch_overlap', default=(64,64,64))
    parser.add_argument('-b', '--batch_size', default=1, type=int)
    parser.add_argument('-n', '--network', default='unet3d')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--labeled_ratio', type=int, default=1, help='1/labeled_ratio data is provided mask')
    parser.add_argument("--kernel_size", default=9, type=int, help="kernel size")  # 9 refers to 1*9/9*1 for DSConv
    parser.add_argument("--extend_scope", default=1.0, type=float, help="extend scope")  # This parameter is not used
    parser.add_argument("--if_offset", default=True, type=bool, help="if offset")  # Whether to use the deformation or not
    parser.add_argument("--n_basic_layer", default=8, type=int, help="basic layer numbers")
    parser.add_argument("--dim", default=8, type=int, help="dim numbers")
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
    # model = UXNET(
    #     in_chans= cfg['IN_CHANNELS'],
    #     out_chans=cfg['NUM_CLASSES'],
    #     depths=[2, 2, 2, 2],
    #     feat_size=[48, 96, 192, 384],
    #     drop_path_rate=0,
    #     layer_scale_init_value=1e-6,
    #     spatial_dims=3,
    # )
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Model
    # model = DSCNet(
    #     n_channels=cfg['IN_CHANNELS'],
    #     n_classes= cfg['NUM_CLASSES'],
    #     kernel_size=args.kernel_size,
    #     extend_scope=args.extend_scope,
    #     if_offset=args.if_offset,
    #     device=device,
    #     number=args.n_basic_layer,
    #     dim=args.dim,
    # )
    # Model
    model = get_network(args.network, cfg['IN_CHANNELS'], cfg['NUM_CLASSES'],img_shape=args.patch_size)
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
    val_path = os.path.join(args.divided, val)
    avg_metric= test_all_case(model, root_path,method=args.network, test_list=val_path, num_classes=num_classes,
                                           patch_size=args.patch_size, stride_xy=64, stride_z=64,test_save_path=test_save_path)
    
    print(avg_metric)
