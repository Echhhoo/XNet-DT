import numpy as np
import os
import argparse
from tqdm import tqdm
import SimpleITK as sitk

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='/home/ac/datb/wch/imagecas')
    parser.add_argument('--save_path', default='/home/ac/datb/wch/imagecas_part')
    args = parser.parse_args()
    output_size = [128,128,128]
    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)
    save_image_path = args.save_path + '/img'
    save_mask_path = args.save_path + '/mask'
    if not os.path.exists(save_image_path):
        os.mkdir(save_image_path)
    if not os.path.exists(save_mask_path):
        os.mkdir(save_mask_path)

    image_path = args.data_path + '/img'
    mask_path = args.data_path + '/mask'

    for i in os.listdir(image_path):

        image_dir = os.path.join(image_path, i)
        mask_dir = os.path.join(mask_path, i.replace("img", "mask"))

        image = sitk.ReadImage(image_dir)
        mask = sitk.ReadImage(mask_dir)

        size = np.array(image.GetSize())
        spacing = np.array(image.GetSpacing())
        target_spacing=[spacing[0],spacing[0],spacing[0]]
        new_size = size * spacing / target_spacing
        new_size = [int(s) for s in new_size]
        
        print(i,image.GetSpacing(),size,new_size)
 
        resample_image = sitk.ResampleImageFilter()
        resample_image.SetOutputDirection(image.GetDirection())
        resample_image.SetOutputOrigin(image.GetOrigin())
        resample_image.SetSize(new_size)
        resample_image.SetOutputSpacing(target_spacing)
        resample_image.SetInterpolator(sitk.sitkLinear)
        image = resample_image.Execute(image)

        resample_mask = sitk.ResampleImageFilter()
        resample_mask.SetOutputDirection(mask.GetDirection())
        resample_mask.SetOutputOrigin(mask.GetOrigin())
        resample_mask.SetSize(new_size)
        resample_mask.SetOutputSpacing(target_spacing)
        resample_mask.SetInterpolator(sitk.sitkNearestNeighbor)
        mask = resample_mask.Execute(mask)

        image_np = sitk.GetArrayFromImage(image)
        mask_np = sitk.GetArrayFromImage(mask)
        mask_np1 = sitk.GetArrayFromImage(mask)
        mask_np1= (mask_np1 != 0).astype(np.uint8)
        # image_np= image_np.transpose(2, 1, 0)
        # mask_np  = mask_np.transpose(2, 1, 0)
        # mask_np1  = mask_np1.transpose(2, 1, 0)
        cut_size=image_np.shape


        w, h, d = mask_np1.shape

        tempL = np.nonzero(mask_np1)  # 得到非零函数的位置，也就是说这里得到的是标签像素的位置
        minx, maxx = np.min(tempL[0]), np.max(tempL[0])
        miny, maxy = np.min(tempL[1]), np.max(tempL[1])
        minz, maxz = np.min(tempL[2]), np.max(tempL[2])

        px = max(output_size[0] - (maxx - minx), 0) // 2
        py = max(output_size[1] - (maxy - miny), 0) // 2
        pz = max(output_size[2] - (maxz - minz), 0) // 2
        minx = max(minx - np.random.randint(5, 10) - px, 0)
        maxx = min(maxx + np.random.randint(5, 10) + px, w)
        miny = max(miny - np.random.randint(5, 10) - py, 0)
        maxy = min(maxy + np.random.randint(5, 10) + py, h)
        minz = max(minz - np.random.randint(5, 10) - pz, 0)
        maxz = min(maxz + np.random.randint(5, 10) + pz, d)
         # 判断图像尺寸的三个维度是否为偶数，如果不是偶数则删除对应维度最后一个切片
        if (maxx-minx) % 2!= 0:
            maxx+=1
        if (maxy-miny) % 2!= 0:
            maxy+=1
        if (maxz-minz) % 2!= 0:
            maxz+=1

        image_np = image_np[minx:maxx, miny:maxy, minz:maxz]
        mask_np = mask_np[minx:maxx, miny:maxy, minz:maxz]
        


        image_save = sitk.GetImageFromArray(image_np)
        image_save.SetSpacing(target_spacing)
        image_save.SetDirection(image.GetDirection())
        image_save.SetOrigin(image.GetOrigin())

        mask_save = sitk.GetImageFromArray(mask_np)
        mask_save.SetSpacing(target_spacing)
        mask_save.SetDirection(image.GetDirection())
        mask_save.SetOrigin(image.GetOrigin())
        print('cut:',cut_size,image_save.GetSize())

        sitk.WriteImage(image_save, os.path.join(save_image_path, i))
        sitk.WriteImage(mask_save, os.path.join(save_mask_path, i))
    
