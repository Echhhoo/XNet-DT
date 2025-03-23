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
    parser.add_argument('--image_path', default='/home/ac/datb/wch/imagecass/img')
    parser.add_argument('--mask_path', default='/home/ac/datb/wch/imagecass/mask')
    parser.add_argument('--DTCL_path', default='/home/ac/datb/wch/imagecass/DTCL')
    parser.add_argument('--DTCH_path', default='/home/ac/datb/wch/imagecass/DTCH')
    parser.add_argument('--DTCHA_path', default='/home/ac/datb/wch/imagecass/DTCHA')
    parser.add_argument('--DTCHS_path', default='/home/ac/datb/wch/imagecass/DTCHS')
    parser.add_argument('--DTCLHA_path', default='/home/ac/datb/wch/imagecass/DTCLHA')
    parser.add_argument('--DTCLHS_path', default='/home/ac/datb/wch/imagecass/DTCLHS')


    parser.add_argument('--dtcwavelet_type', default=1, help='1,2,3,4')

    args = parser.parse_args()

    for i in os.listdir(args.image_path):
        image_path = os.path.join(args.image_path, i)
        image = sitk.ReadImage(image_path)
        print(i,image.GetSize())
        image_np = sitk.GetArrayFromImage(image)
        # image_np = (image_np - image_np.min()) / (image_np.max() - image_np.min()) * 255 
        
        # mask_path = os.path.join(args.mask_path, i)
        # mask = sitk.ReadImage(mask_path)
        # mask_np = sitk.GetArrayFromImage(mask)
        
         # 创建一个 SimpleITK.ResampleImageFilter 对象，用于对图像进行重采样操作。
        resample_image = sitk.ResampleImageFilter()
        resample_image.SetSize(image.GetSize()) #设置重采样后的图像大小为 image 的大小
        resample_image.SetOutputSpacing([0.5, 0.5, 0.5]) #设置重采样后的图像像素间距为 [0.5, 0.5, 0.5]
        resample_image.SetInterpolator(sitk.sitkLinear) #设置重采样时使用的插值方法为线性插值
        
        # if image_np.shape[0] % 2!= 0:
        #     # padding = (0, 1) if image_np.shape[0] % 2 == 1 else (0, 0)
        #     # image_np = np.pad(image_np, (padding,(0, 0), (0, 0)), mode='constant')
        #     # mask_np = np.pad(mask_np, (padding,(0, 0), (0, 0)), mode='constant')
        #     image_np = image_np[:image_np.shape[0] - 1,...]
        #     mask_np = mask_np[:mask_np.shape[0] - 1,...]
            
            
        #     mask = sitk.GetImageFromArray(mask_np)
            
            
        # Specify number of levels and wavelet family to use
        nlevels = 1
        b = biort('near_sym_a')
        q = qshift('qshift_a')

        # Form the DT-CWT of the sphere. We use discard_level_1 since we're
        # Yl, Yh = dtwavexfm3(image_np, nlevels, b, q, discard_level_1=True)
        # Plot maxima 
        
        transform = dtcwt.Transform3d()
        mandrill_t = transform.forward(image_np, nlevels=1) 
       
        LLL=mandrill_t.lowpass
        # 对 LLL 进行归一化操作，将值映射到 [0, 255] 的范围
        LLL = (LLL - LLL.min()) / (LLL.max() - LLL.min()) * 255 
        # LLL = LLL[::2, ::2, ::2]
       
        high=[]
        DTCH=np.zeros((mandrill_t.highpasses[0].shape[0], mandrill_t.highpasses[0].shape[1],mandrill_t.highpasses[0].shape[2]))
        # DTCHA=np.zeros((mandrill_t.highpasses[0].shape[0], mandrill_t.highpasses[0].shape[1],mandrill_t.highpasses[0].shape[2]))
        # DTCHS=np.zeros((mandrill_t.highpasses[0].shape[0], mandrill_t.highpasses[0].shape[1],mandrill_t.highpasses[0].shape[2]))
        for item in range(0,mandrill_t.highpasses[0].shape[3]):
            high.append(np.abs(mandrill_t.highpasses[0][:,:,:,item]))
            high[item]=(high[item]-high[item].min())/(high[item].max()-high[item].min())*255
        #     # if(item<mandrill_t.highpasses[0].shape[3]/2):
        #     #     DTCHA+=np.abs(mandrill_t.highpasses[0][:,:,:,item])
        #     # else:
        #     #     DTCHS+=np.abs(mandrill_t.highpasses[0][:,:,:,item])
            DTCH+=np.abs(mandrill_t.highpasses[0][:,:,:,item])
        # DTCLHA=DTCHA+LLL
        # DTCLHS=DTCHS+LLL
        
        LLL = sitk.GetImageFromArray(LLL)
        
        # LLL = resample_image.Execute(LLL) #执行重采样操作，将重采样后的结果重新赋值给 LLL
        LLL.SetSpacing(image.GetSpacing()) #设置 LLL 的像素间距为 image 的像素间距
        LLL.SetDirection(image.GetDirection()) #设置 LLL 的方向为 image 的方向
        LLL.SetOrigin(image.GetOrigin()) #设置 LLL 的原点为 image 的原点
        sitk.WriteImage(LLL,os.path.join(args.DTCL_path, i))
       
        

        # LLL_np = sitk.GetArrayFromImage(LLL)
        # DTCHA = sitk.GetImageFromArray(DTCHA)
        
        # DTCHA = resample_image.Execute(DTCHA) #执行重采样操作，将重采样后的结果重新赋值给 DTCHA
        # DTCHA.SetSpacing(image.GetSpacing()) #设置 DTCHA 的像素间距为 image 的像素间距
        # DTCHA.SetDirection(image.GetDirection()) #设置 DTCHA 的方向为 image 的方向
        # DTCHA.SetOrigin(image.GetOrigin()) #设置 DTCHA 的原点为 image 的原点
        
        # sitk.WriteImage(DTCHA,os.path.join(args.DTCHA_path, i))
        
        
        # DTCHS = sitk.GetImageFromArray(DTCHS)
        
        # DTCHS = resample_image.Execute(DTCHS) #执行重采样操作，将重采样后的结果重新赋值给 DTCHS
        # DTCHS.SetSpacing(image.GetSpacing()) #设置 DTCHS 的像素间距为 image 的像素间距
        # DTCHS.SetDirection(image.GetDirection()) #设置 DTCHS 的方向为 image 的方向
        # DTCHS.SetOrigin(image.GetOrigin()) #设置 DTCHS 的原点为 image 的原点
        
        # sitk.WriteImage(DTCHS,os.path.join(args.DTCHS_path, i))
        
        
        DTCH = sitk.GetImageFromArray(DTCH)
        
        DTCH = resample_image.Execute(DTCH) #执行重采样操作，将重采样后的结果重新赋值给 DTCH
        DTCH.SetSpacing(image.GetSpacing()) #设置 DTCH 的像素间距为 image 的像素间距
        DTCH.SetDirection(image.GetDirection()) #设置 DTCH 的方向为 image 的方向
        DTCH.SetOrigin(image.GetOrigin()) #设置 DTCH 的原点为 image 的原点
        
        sitk.WriteImage(DTCH,os.path.join(args.DTCH_path, i))
        
        # DTCLHA = sitk.GetImageFromArray(DTCLHA)
        
        # DTCLHA = resample_image.Execute(DTCLHA) #执行重采样操作，将重采样后的结果重新赋值给 DTCLHA
        # DTCLHA.SetSpacing(image.GetSpacing()) #设置 DTCLHA 的像素间距为 image 的像素间距
        # DTCLHA.SetDirection(image.GetDirection()) #设置 DTCLHA 的方向为 image 的方向
        # DTCLHA.SetOrigin(image.GetOrigin()) #设置 DTCLHA 的原点为 image 的原点
        
        # sitk.WriteImage(DTCLHA,os.path.join(args.DTCLHA_path, i))
        
        # DTCLHS = sitk.GetImageFromArray(DTCLHS)
        
        # DTCLHS = resample_image.Execute(DTCLHS) #执行重采样操作，将重采样后的结果重新赋值给 DTCLHS
        # DTCLHS.SetSpacing(image.GetSpacing()) #设置 DTCLHS 的像素间距为 image 的像素间距
        # DTCLHS.SetDirection(image.GetDirection()) #设置 DTCLHS 的方向为 image 的方向
        # DTCLHS.SetOrigin(image.GetOrigin()) #设置 DTCLHS 的原点为 image 的原点
        
        # sitk.WriteImage(DTCLHS,os.path.join(args.DTCLHS_path, i))










