import numpy as np
from glob import glob
from tqdm import tqdm
import h5py
import SimpleITK as sitk
import os
import h5py
output_size = [128, 128, 128]#256是针对uxnet的预处理，常规的是调整为224，224，160，然后patch为112，112，80
def covert_h5():
    img_list = glob('/home/ac/datb/wch/imagecas_part/img/*.nii.gz')
    # min_hu=-100
    # max_hu=600
    for item in tqdm(img_list):
        
        print(item)
        image = sitk.ReadImage(item)
        
        mask_path=item.replace("img/", "mask/")
        if os.path.exists(mask_path):
            mask = sitk.ReadImage(mask_path)
         
        # size = np.array(image.GetSize())
        # spacing = np.array(image.GetSpacing())
        # new_size = size * spacing / [0.5, 0.5, 0.5]
        # new_size = [int(s) for s in new_size]

        # print(new_size, size)

        # resample_image = sitk.ResampleImageFilter()
        # resample_image.SetOutputDirection(image.GetDirection())
        # resample_image.SetOutputOrigin(image.GetOrigin())
        # resample_image.SetSize(new_size)
        # resample_image.SetOutputSpacing([0.5, 0.5, 0.5])
        # resample_image.SetInterpolator(sitk.sitkLinear)
        # image = resample_image.Execute(image)
        
        # resample_mask = sitk.ResampleImageFilter()
        # resample_mask.SetOutputDirection(mask.GetDirection())
        # resample_mask.SetOutputOrigin(mask.GetOrigin())
        # resample_mask.SetSize(new_size)
        # resample_mask.SetOutputSpacing([0.5, 0.5, 0.5])
        # resample_mask.SetInterpolator(sitk.sitkNearestNeighbor)
        # mask = resample_mask.Execute(mask)

       
       
        image_np= sitk.GetArrayFromImage(image)
        label1_np = sitk.GetArrayFromImage(mask)
        # label2_np = sitk.GetArrayFromImage(mask)
        # label2_np = (label2_np != 0).astype(np.uint8)
        # print(np.unique(label1_np))
        image_np = image_np.transpose(2, 1, 0)
        label_np = label1_np.transpose(2, 1, 0)
        # label2_np = label2_np.transpose(2, 1, 0)
       
        # w, h, d = label2_np.shape

        # tempL = np.nonzero(label2_np)  # 得到非零函数的位置，也就是说这里得到的是标签像素的位置
        # minx, maxx = np.min(tempL[0]), np.max(tempL[0])
        # miny, maxy = np.min(tempL[1]), np.max(tempL[1])
        # minz, maxz = np.min(tempL[2]), np.max(tempL[2])

        # px = max(output_size[0] - (maxx - minx), 0) // 2
        # py = max(output_size[1] - (maxy - miny), 0) // 2
        # pz = max(output_size[2] - (maxz - minz), 0) // 2
        # minx = max(minx - np.random.randint(10, 20) - px, 0)
        # maxx = min(maxx + np.random.randint(10, 20) + px, w)
        # miny = max(miny - np.random.randint(10, 20) - py, 0)
        # maxy = min(maxy + np.random.randint(10, 20) + py, h)
        # minz = max(minz - np.random.randint(5, 10) - pz, 0)
        # maxz = min(maxz + np.random.randint(5, 10) + pz, d)

        # image_np = image_np[minx:maxx, miny:maxy, minz:maxz]
        # label_np = label1_np[minx:maxx, miny:maxy, minz:maxz]
        
        # image_np[image_np < min_hu] = min_hu
        # image_np[image_np > max_hu] = max_hu

        image_np = (image_np - np.mean(image_np)) / np.std(image_np)
        image_np = image_np.astype(np.float32)
        # image_re= sitk.GetImageFromArray(image_np)


        f = h5py.File(item.replace('.nii.gz', '_norm.h5').replace("/imagecas_part/img/", "/imagecas_h5/"), 'w')
        f.create_dataset('image', data=image_np, compression="gzip")
        f.create_dataset('label', data=label_np, compression="gzip")
        print(f['image'][:].shape)
        print(f['label'][:].shape)
        f.close()
        # sitk.WriteImage(image_re,path)
if __name__ == '__main__':
    covert_h5()