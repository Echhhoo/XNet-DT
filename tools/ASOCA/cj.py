import SimpleITK as sitk
import argparse
import os
import numpy as np
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # parser.add_argument('--image_path', default='//10.0.5.233/shared_data/XNet/dataset/LiTS/val/image')
    # parser.add_argument('--L_path', default='//10.0.5.233/shared_data/XNet/dataset/LiTS/train_sup_100/L')
    # parser.add_argument('--H_path', default='//10.0.5.233/shared_data/XNet/dataset/LiTS/train_sup_100/H')
    parser.add_argument('--image_path', default='/home/ac/datb/wch/imagecas_part/img')
    parser.add_argument('--mask_path', default='/home/ac/datb/wch/imagecas_part/mask')
    args = parser.parse_args()

    for i in os.listdir(args.image_path):
        print(i)
        image = sitk.ReadImage(os.path.join(args.image_path, i))
        # 加载nii.gz格式的图像
        image_np = sitk.GetArrayFromImage(image)
        print(image_np.shape)
        
        mask = sitk.ReadImage(os.path.join(args.mask_path, i))
        mask_np = sitk.GetArrayFromImage(mask)
        
        if image_np.shape[1] % 2!= 0:
            print("crop!")
            image_np= image_np.transpose(2, 1, 0)
            mask_np  = mask_np.transpose(2, 1, 0)
             # 假设在第一个维度进行裁剪，这里裁剪掉最后1个切片（可根据实际需求修改）
            cropped_array = image_np[:,:-1, :]
            cropped_image = sitk.GetImageFromArray(cropped_array)

            # 设置裁剪后图像的空间信息（方向、原点、间距）与原始图像相同
            cropped_image.SetDirection(image.GetDirection())
            cropped_image.SetOrigin(image.GetOrigin())
            cropped_image.SetSpacing(image.GetSpacing())
            
            size = np.array(cropped_image.GetSize())
            print(size)
            # 保存裁剪后的图像
            sitk.WriteImage(cropped_image, os.path.join(args.image_path, i))
            
            cropped_array = mask_np[:, :-1, :]
            cropped_image = sitk.GetImageFromArray(cropped_array)

            # 设置裁剪后图像的空间信息（方向、原点、间距）与原始图像相同
            cropped_image.SetDirection(image.GetDirection())
            cropped_image.SetOrigin(image.GetOrigin())
            cropped_image.SetSpacing(image.GetSpacing())

            # 保存裁剪后的图像
            sitk.WriteImage(cropped_image, os.path.join(args.mask_path, i))
       
        
    
