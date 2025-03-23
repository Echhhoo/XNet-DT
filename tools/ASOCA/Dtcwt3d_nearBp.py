import numpy as np
from PIL import Image
import pywt
import argparse
import os
import SimpleITK as sitk
from pytorch_wavelets import DTCWTForward, DTCWTInverse
from einops import rearrange
import dtcwt
from dtcwt.compat import dtwavexfm3, dtwaveifm3
from dtcwt.coeffs import biort, qshift

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # parser.add_argument('--image_path', default='//10.0.5.233/shared_data/XNet/dataset/LiTS/val/image')
    # parser.add_argument('--L_path', default='//10.0.5.233/shared_data/XNet/dataset/LiTS/train_sup_100/L')
    # parser.add_argument('--H_path', default='//10.0.5.233/shared_data/XNet/dataset/LiTS/train_sup_100/H')
    parser.add_argument('--root_path', default='/home/ac/datb/wch/')
    parser.add_argument('--dataset', default='ASOCA_part')
    parser.add_argument('--dtcwavelet_type', default=1, help='1,2,3,4')

    args = parser.parse_args()
    path=args.root_path+args.dataset+'/img/'
    for i in os.listdir(path):
        
        image_path = os.path.join(path, i)
        image = sitk.ReadImage(image_path)
        spacing = np.array(image.GetSpacing())
        target_spacing=[spacing[0],spacing[0],spacing[0]]
        ori_spacing=[spacing[0]*2,spacing[0]*2,spacing[0]*2]
     
        print(i,image.GetSize())
        image_np = sitk.GetArrayFromImage(image)
   
        
         # 创建一个 SimpleITK.ResampleImageFilter 对象，用于对图像进行重采样操作。
        resample_image = sitk.ResampleImageFilter()
        resample_image.SetSize(image.GetSize()) #设置重采样后的图像大小为 image 的大小
        resample_image.SetOutputSpacing(target_spacing) #设置重采样后的图像像素间距为 [0.5, 0.5, 0.5]
        resample_image.SetInterpolator(sitk.sitkLinear) #设置重采样时使用的插值方法为线性插值
        
            
            
        # Specify number of levels and wavelet family to use
        nlevels = 1
        b = biort('near_sym_b_bp')
        q = qshift('qshift_a')
        
        # transform = dtcwt.Transform3d(b, q)
        # mandrill_t = transform.forward(image_np, nlevels=1) 
        # Yl=mandrill_t.lowpass
        # Yh=mandrill_t.highpasses

        # # Form the DT-CWT of the sphere. We use discard_level_1 since we're
        # Yl, Yh = dtwavexfm3(image_np, nlevels, b, q, discard_level_1=False)
        transform = dtcwt.Transform3d(b,q)
        mandrill_t = transform.forward(image_np, nlevels=1) 
        Yl=mandrill_t.lowpass
        Yh=mandrill_t.highpasses
        # Plot maxima 
        

        # 对 LLL 进行归一化操作，将值映射到 [0, 255] 的范围
        LLL = (Yl - Yl.min()) / (Yl.max() - Yl.min()) * 255 
        
        
        DTCH =np.zeros((Yh[-1].shape[0], Yh[-1].shape[1], Yh[-1].shape[2]))
        DTCHA=np.zeros((Yh[-1].shape[0], Yh[-1].shape[1], Yh[-1].shape[2]))
        DTCHR=np.zeros((Yh[-1].shape[0], Yh[-1].shape[1], Yh[-1].shape[2]))
        high=[]
        for idx in range(Yh[-1].shape[3]):
            high.append(np.abs(Yh[-1][:, :, :, idx]))
            high[idx]=(high[idx]-high[idx].min())/(high[idx].max()-high[idx].min())*255
            if(0<=idx<8 or 12<=idx<16):
                DTCHA+=np.abs(high[idx])
            else:
                DTCHR+=np.abs(high[idx])
            DTCH+=np.abs(high[idx])
        DTCHA= (DTCHA-  DTCHA.min()) / (DTCHA.max() - DTCHA.min()) * 255
        DTCHR =(DTCHR - DTCHR.min()) / (DTCHR.max() - DTCHR.min()) * 255
        DTCH = (DTCH- DTCH.min()) / (DTCH.max() - DTCH.min()) * 255
        LLL = sitk.GetImageFromArray(LLL)
        
        # LLL = resample_image.Execute(LLL) #执行重采样操作，将重采样后的结果重新赋值给 LLL
        LLL.SetSpacing(image.GetSpacing()) #设置 LLL 的像素间距为 image 的像素间距
        LLL.SetDirection(image.GetDirection()) #设置 LLL 的方向为 image 的方向
        LLL.SetOrigin(image.GetOrigin()) #设置 LLL 的原点为 image 的原点
        DTCL_path=args.root_path+args.dataset+'/P_DTCL/'
        sitk.WriteImage(LLL,os.path.join(DTCL_path, i))
       
        
        DTCHA = sitk.GetImageFromArray(DTCHA)
        DTCHA.SetSpacing(ori_spacing)
        DTCHA = resample_image.Execute(DTCHA) #执行重采样操作，将重采样后的结果重新赋值给 DTCHA
        DTCHA.SetSpacing(image.GetSpacing()) #设置 DTCHA 的像素间距为 image 的像素间距
        DTCHA.SetDirection(image.GetDirection()) #设置 DTCHA 的方向为 image 的方向
        DTCHA.SetOrigin(image.GetOrigin()) #设置 DTCHA 的原点为 image 的原点
        DTCHA_path=args.root_path+args.dataset+'/P_DTCHA/'
        sitk.WriteImage(DTCHA,os.path.join(DTCHA_path, i))
        
        
        DTCHR = sitk.GetImageFromArray(DTCHR)
        DTCHR.SetSpacing(ori_spacing)
        DTCHR= resample_image.Execute(DTCHR) #执行重采样操作，将重采样后的结果重新赋值给 DTCHS
        DTCHR.SetSpacing(image.GetSpacing()) #设置 DTCHS 的像素间距为 image 的像素间距
        DTCHR.SetDirection(image.GetDirection()) #设置 DTCHS 的方向为 image 的方向
        DTCHR.SetOrigin(image.GetOrigin()) #设置 DTCHS 的原点为 image 的原点
        DTCHR_path=args.root_path+args.dataset+'/P_DTCHR/'
        sitk.WriteImage(DTCHR,os.path.join(DTCHR_path, i))
        
        
        # DTCH = sitk.GetImageFromArray(DTCH)
        # DTCH.SetSpacing(ori_spacing)
        # DTCH = resample_image.Execute(DTCH) #执行重采样操作，将重采样后的结果重新赋值给 DTCH
        # DTCH.SetSpacing(image.GetSpacing()) #设置 DTCH 的像素间距为 image 的像素间距
        # DTCH.SetDirection(image.GetDirection()) #设置 DTCH 的方向为 image 的方向
        # DTCH.SetOrigin(image.GetOrigin()) #设置 DTCH 的原点为 image 的原点
        # DTCH_path=args.root_path+args.dataset+'/DTCH/'
        # sitk.WriteImage(DTCH,os.path.join(DTCH_path, i))
        








