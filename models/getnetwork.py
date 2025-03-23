import sys
from models import *
from models.networks_3d.XNetv2 import xnetv2_3d,xnetv2_3d_min
from models.networks_3d.XNetv3 import xnetv3_3d_min,xnetv3_3d_Add,xnetv3_3D_cat1,xnetv3_3D_add0,xnetv3_3D_add1,xnetv3_3d_1cat,xnetv3_3d_0cat
import torch.nn as nn

def get_network(network, in_channels, num_classes, **kwargs):

    # 2d networks
    if network == 'xnet':
        net = XNet(in_channels, num_classes)
    elif network == 'xnet_sb':
        net = XNet_sb(in_channels, num_classes)
    elif network == 'xnet_1_1_m':
        net = XNet_1_1_m(in_channels, num_classes)
    elif network == 'xnet_1_2_m':
        net = XNet_1_2_m(in_channels, num_classes)
    elif network == 'xnet_2_1_m':
        net = XNet_2_1_m(in_channels, num_classes)
    elif network == 'xnet_3_2_m':
        net = XNet_3_2_m(in_channels, num_classes)
    elif network == 'xnet_2_3_m':
        net = XNet_2_3_m(in_channels, num_classes)
    elif network == 'xnet_3_3_m':
        net = XNet_3_3_m(in_channels, num_classes)
    elif network == 'unet':
        net = unet(in_channels, num_classes)
    elif network == 'unet_plusplus' or network == 'unet++':
        net = unet_plusplus(in_channels, num_classes)
    elif network == 'r2unet':
        net = r2_unet(in_channels, num_classes)
    elif network == 'attunet':
        net = attention_unet(in_channels, num_classes)
    elif network == 'hrnet18':
        net = hrnet18(in_channels, num_classes)
    elif network == 'hrnet48':
        net = hrnet48(in_channels, num_classes)
    elif network == 'resunet':
        net = res_unet(in_channels, num_classes)
    elif network == 'resunet++':
        net = res_unet_plusplus(in_channels, num_classes)
    elif network == 'u2net':
        net = u2net(in_channels, num_classes)
    elif network == 'u2net_s':
        net = u2net_small(in_channels, num_classes)
    elif network == 'unet3+':
        net = unet_3plus(in_channels, num_classes)
    elif network == 'unet3+_ds':
        net = unet_3plus_ds(in_channels, num_classes)
    elif network == 'unet3+_ds_cgm':
        net = unet_3plus_ds_cgm(in_channels, num_classes)
    elif network == 'swinunet':
        net = swinunet(num_classes, 224)  # img_size = 224
    elif network == 'unet_cct':
        net = unet_cct(in_channels, num_classes)
    elif network == 'wavesnet':
        net = wsegnet_vgg16_bn(in_channels, num_classes)
    elif network == 'mwcnn':
        net = mwcnn(in_channels, num_classes)
    elif network == 'alnet':
        net = Aerial_LaneNet(in_channels, num_classes)
    elif network == 'wds':
        net = WDS(in_channels, num_classes)

    # 3d networks
    elif network == 'xnet3d':
        net = xnet3d(in_channels, num_classes)
    elif network == 'XNetv2' or network == 'xnetv2':
        net = xnetv2_3d(in_channels, num_classes)
    elif network == 'XNetv2_3D_min' or network == 'xnetv2_3d_min':
        net = xnetv2_3d_min(in_channels, num_classes)
    # elif network == 'XNetv3' or network == 'xnetv3':
    #     net = xnetv3_3d(in_channels, num_classes)
    elif network == 'XNetv3_3D_min' or network == 'xnetv3_3d_min':
        net = xnetv3_3d_min(in_channels, num_classes)
    # elif network == 'XNetv3_3D_new' or network == 'xnetv3_3d_new':
    #     net = xnetv3_3d_new(in_channels, num_classes)
    # elif network == 'XNetv3_3D_right' or network == 'xnetv3_3d_right':
    #     net = xnetv3_3d_right(in_channels, num_classes)
    # elif network == 'XNetv3_3D_xh' or network == 'xnetv3_3d_xh':
    #     net = xnetv3_3d_xh(in_channels, num_classes)
    # elif network == 'XNetv3_3D_jh' or network == 'xnetv3_3d_jh':
    #     net = xnetv3_3d_jh(in_channels, num_classes)
    # elif network == 'XNetv3_3D_re' or network == 'xnetv3_3d_re':
        # net = xnetv3_3d_re(in_channels, num_classes)
    elif network == 'XNetv3_3D_Add' or network == 'xnetv3_3d_add':
        net = xnetv3_3d_Add(in_channels, num_classes)
    elif network == 'XNetv3_3D_0cat' or network == 'xnetv3_3d_0cat':
        net = xnetv3_3d_0cat(in_channels, num_classes)
    elif network == 'XNetv3_3D_1cat' or network == 'xnetv3_3d_1cat':
        net = xnetv3_3d_1cat(in_channels, num_classes)
        
    elif network == 'xnetv3_3D_CAT1' or network == 'xnetv3_3d_cat1':
        net = xnetv3_3D_cat1(in_channels, num_classes)
    elif network == 'XNetv3_3D_Add0' or network == 'xnetv3_3d_add0':
        net = xnetv3_3D_add0(in_channels, num_classes)
    elif network == 'XNetv3_3D_Add1' or network == 'xnetv3_3d_add1':
        net = xnetv3_3D_add1(in_channels, num_classes)
        
        
    elif network == 'unet3d':
        net = unet3d(in_channels, num_classes)
    elif network == 'unet3d_min':
        net = unet3d_min(in_channels, num_classes)
    # elif network == 'unet3d_ssl':
    #     net =  unet3d_ssl(in_channels, num_classes)
    elif network == 'unet3d_urpc':
        net = unet3d_urpc(in_channels, num_classes)
    elif network == 'unet_urpc':
        net = unet_urpc(in_channels, num_classes)
    elif network == 'unet3d_cct':
        net = unet3d_cct(in_channels, num_classes)
    elif network == 'unet3d_cct_min':
        net = unet3d_cct_min(in_channels, num_classes)
    elif network == 'unet3d_dtc':
        net = unet3d_dtc(in_channels, num_classes)
    elif network == 'vnet':
        net = vnet(in_channels, num_classes)
    elif network == 'vnet_cct':
        net = vnet_cct(in_channels, num_classes)
    elif network == 'vnet_dtc':
        net = vnet_dtc(in_channels, num_classes)
    elif network == 'resunet3d':
        net = res_unet3d(in_channels, num_classes)
    elif network == 'conresnet':
        net = conresnet(in_channels, num_classes, img_shape=kwargs['img_shape'])
    elif network == 'espnet3d':
        net = espnet3d(in_channels, num_classes)
    elif network == 'dmfnet':
        net = dmfnet(in_channels, num_classes)
    elif network == 'transbts':
        net = transbts(in_channels, num_classes, img_shape=kwargs['img_shape'])
    elif network == 'cotr':
        net = cotr(in_channels, num_classes)
    elif network == 'unertr':
        net = unertr(in_channels, num_classes, img_shape=kwargs['img_shape'])
    else:
        print('the network you have entered is not supported yet')
        sys.exit()
    return net
