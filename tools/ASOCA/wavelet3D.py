import numpy as np
from PIL import Image
import pywt
import argparse
import os
import SimpleITK as sitk
import torchio as tio

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # parser.add_argument('--image_path', default='//10.0.5.233/shared_data/XNet/dataset/LiTS/val/image')
    # parser.add_argument('--L_path', default='//10.0.5.233/shared_data/XNet/dataset/LiTS/train_sup_100/L')
    # parser.add_argument('--H_path', default='//10.0.5.233/shared_data/XNet/dataset/LiTS/train_sup_100/H')
    parser.add_argument('--image_path', default='/home/ac/datb/wch/imagecas_part/img')
    parser.add_argument('--L_path', default='/home/ac/datb/wch/imagecas_part/L')
    parser.add_argument('--H_path', default='/home/ac/datb/wch/imagecas_part/H')
    parser.add_argument('--wavelet_type', default='db2', help='haar, db2, bior1.5, bior2.4, coif1, dmey')

    args = parser.parse_args()

    if not os.path.exists(args.L_path):
        os.mkdir(args.L_path)

    if not os.path.exists(args.H_path):
        os.mkdir(args.H_path)

    for i in os.listdir(args.image_path):

        image_path = os.path.join(args.image_path, i)
        
        L_path = os.path.join(args.L_path, i)
        H_path = os.path.join(args.H_path, i)

        image = sitk.ReadImage(image_path)
        print(image_path,image.GetSize())
        image_np = sitk.GetArrayFromImage(image)
       
        image_wave = pywt.dwtn(image_np, args.wavelet_type)
        spacing = np.array(image.GetSpacing())
        target_spacing=[spacing[0],spacing[0],spacing[0]]
        ori_spacing=[spacing[0]*2,spacing[0]*2,spacing[0]*2]
     
        LLL = image_wave['aaa']
        LLH = image_wave['aad']
        LHL = image_wave['ada']
        LHH = image_wave['add']
        HLL = image_wave['daa']
        HLH = image_wave['dad']
        HHL = image_wave['dda']
        HHH = image_wave['ddd']
        # 对 LLL 进行归一化操作，将值映射到 [0, 255] 的范围
        LLL = (LLL - LLL.min()) / (LLL.max() - LLL.min()) * 255 
        LLL = sitk.GetImageFromArray(LLL)
        
        # 创建一个 SimpleITK.ResampleImageFilter 对象，用于对图像进行重采样操作。
        resample_image = sitk.ResampleImageFilter()
        resample_image.SetSize(image.GetSize()) #设置重采样后的图像大小为 image 的大小
        resample_image.SetOutputSpacing(target_spacing) #设置重采样后的图像像素间距为 [0.5, 0.5, 0.5]
        resample_image.SetInterpolator(sitk.sitkLinear) #设置重采样时使用的插值方法为线性插值
        
        LLL.SetSpacing(ori_spacing) #设置 LLL 的像素间距为 image 的像素间距
        LLL = resample_image.Execute(LLL) #执行重采样操作，将重采样后的结果重新赋值给 LLL
        

       
        LLL.SetDirection(image.GetDirection()) #设置 LLL 的方向为 image 的方向
        LLL.SetOrigin(image.GetOrigin()) #设置 LLL 的原点为 image 的原点
        
        sitk.WriteImage(LLL, L_path)


        LLH = (LLH - LLH.min()) / (LLH.max() - LLH.min()) * 255
        LHL = (LHL - LHL.min()) / (LHL.max() - LHL.min()) * 255
        LHH = (LHH - LHH.min()) / (LHH.max() - LHH.min()) * 255
        HLL = (HLL - HLL.min()) / (HLL.max() - HLL.min()) * 255
        HLH = (HLH - HLH.min()) / (HLH.max() - HLH.min()) * 255
        HHL = (HHL - HHL.min()) / (HHL.max() - HHL.min()) * 255
        HHH = (HHH - HHH.min()) / (HHH.max() - HHH.min()) * 255

        merge1 = LLH + LHL + LHH + HLL + HLH + HHL + HHH
        merge1 = (merge1 - merge1.min()) / (merge1.max() - merge1.min()) * 255

        merge1 = sitk.GetImageFromArray(merge1)

        resample_image = sitk.ResampleImageFilter()
        resample_image.SetSize(image.GetSize())
        resample_image.SetOutputSpacing(target_spacing)
        resample_image.SetInterpolator(sitk.sitkLinear)
        
        merge1.SetSpacing(ori_spacing)
        merge1 = resample_image.Execute(merge1)

        merge1.SetSpacing(target_spacing)
        merge1.SetDirection(image.GetDirection())
        merge1.SetOrigin(image.GetOrigin())

        sitk.WriteImage(merge1, H_path)


