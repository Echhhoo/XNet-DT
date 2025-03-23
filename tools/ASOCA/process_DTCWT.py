import numpy as np
from glob import glob
from tqdm import tqdm
import h5py
import SimpleITK as sitk
import os
import h5py
output_size = [128,128,128]#256是针对uxnet的预处理，常规的是调整为224，224，160，然后patch为112，112，80
def covert_h5():
    img_list = glob('/home/ac/datb/wch/imagecas_part/mask/*.nii.gz')
    for item in tqdm(img_list):
        print(item)
        mask = sitk.ReadImage(item)
        sup=True
        # mask = sitk.ReadImage(mask_itk)
        
        img_path=item.replace("mask", "img")
        image1 = sitk.ReadImage(img_path)
        image1 = sitk.GetArrayFromImage(image1)
        
        image2 = sitk.ReadImage(item.replace("mask", "DTCL"))  
        image2=sitk.GetArrayFromImage(image2)
        
        image3 = sitk.ReadImage(item.replace("mask", "DTCHA"))
        image3=sitk.GetArrayFromImage(image3)
        
        image4 = sitk.ReadImage(item.replace("mask", "DTCHR"))
        image4 = sitk.GetArrayFromImage(image4)
        
        label = sitk.GetArrayFromImage(mask)
        
        # if os.path.exists(item.replace('.nii.gz', '_norm.h5').replace("/imagecas_part/mask/", "/imagecas_DTCWT/")):
        #     print(f"{item} exists, skipping this iteration.")
        #     if (image1.shape == image2.shape and image1.shape == image3.shape and image1.shape == image4.shape and image1.shape == label.shape):
        #         print("所有图像和标签的形状均相同。")
        #     else:
        #         print(image1.shape, "  ", image2.shape, "  ", image3.shape, "  ", image4.shape, "  ", label.shape)
        #     continue  # 使用continue跳过本次循环剩余代码，直接进入下一次循环
            
        # label2 = sitk.GetArrayFromImage(mask)
        # label2 = (label2 != 0).astype(np.uint8)
        # print(np.unique(label1))
        image1 = image1.transpose(2, 1, 0)
        image2 = image2.transpose(2, 1, 0)
        image3 = image3.transpose(2, 1, 0)
        image4 = image4.transpose(2, 1, 0)
        
        label = label.transpose(2, 1, 0)
      
        image1 = (image1 - np.mean(image1)) / np.std(image1)
        image2 = (image2 - np.mean(image2)) / np.std(image2)
        image3 = (image3 - np.mean(image3)) / np.std(image3)
        image4 = (image4 - np.mean(image4)) / np.std(image4)
        image1 = image1.astype(np.float32)
        image2 = image2.astype(np.float32)
        image3 = image3.astype(np.float32)
        image4 = image4.astype(np.float32)

        f = h5py.File(item.replace('.nii.gz', '_norm.h5').replace("/imagecas_part/mask/", "/imagecas_DTCWT/"), 'w')
        f.create_dataset('image1', data=image1, compression="gzip")
        f.create_dataset('image2', data=image2, compression="gzip")
        f.create_dataset('image3', data=image3, compression="gzip")
        f.create_dataset('image4', data=image4, compression="gzip")
        f.create_dataset('label', data=label, compression="gzip")
        print(f['image1'][:].shape)
        print(f['image2'][:].shape)
        print(f['image3'][:].shape)
        print(f['image4'][:].shape)
        print(f['label'][:].shape)
        f.close()
if __name__ == '__main__':
    covert_h5()