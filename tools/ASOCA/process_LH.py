import numpy as np
from glob import glob
from tqdm import tqdm
import h5py
import SimpleITK as sitk
import os
import h5py
output_size = [112, 112, 80]#256是针对uxnet的预处理，常规的是调整为224，224，160，然后patch为112，112，80
def covert_h5():
    img_list = glob('/home/ac/datb/wch/ASOCA_dataset/mask/*.nii.gz')
    for item in tqdm(img_list):
        print(item)
        mask = sitk.ReadImage(item)
        sup=True
        # mask = sitk.ReadImage(mask_itk)
        
        img_path=item.replace("mask", "L")
        image1 = sitk.ReadImage(img_path)
        image1 = sitk.GetArrayFromImage(image1)
        
        image2 = sitk.ReadImage(item.replace("mask", "H"))
        # if os.path.exists(mask_path):
        #     mask = sitk.ReadImage(mask_path)
        # # 如果文件存在，可以在这里进行后续对 mask 的处理
        # else:
        #     sup=False
        
        image2=sitk.GetArrayFromImage(image2)
        label1 = sitk.GetArrayFromImage(mask)
        label2 = sitk.GetArrayFromImage(mask)
        label2 = (label2 != 0).astype(np.uint8)
        print(np.unique(label1))
        image1 = image1.transpose(2, 1, 0)
        image2 = image2.transpose(2, 1, 0)
        label1 = label1.transpose(2, 1, 0)
        label2 = label2.transpose(2, 1, 0)

        w, h, d = label2.shape

        tempL = np.nonzero(label2)  # 得到非零函数的位置，也就是说这里得到的是标签像素的位置
        minx, maxx = np.min(tempL[0]), np.max(tempL[0])
        miny, maxy = np.min(tempL[1]), np.max(tempL[1])
        minz, maxz = np.min(tempL[2]), np.max(tempL[2])

        px = max(output_size[0] - (maxx - minx), 0) // 2
        py = max(output_size[1] - (maxy - miny), 0) // 2
        pz = max(output_size[2] - (maxz - minz), 0) // 2
        minx = max(minx - np.random.randint(10, 20) - px, 0)
        maxx = min(maxx + np.random.randint(10, 20) + px, w)
        miny = max(miny - np.random.randint(10, 20) - py, 0)
        maxy = min(maxy + np.random.randint(10, 20) + py, h)
        minz = max(minz - np.random.randint(5, 10) - pz, 0)
        maxz = min(maxz + np.random.randint(5, 10) + pz, d)

        image1 = image1[minx:maxx, miny:maxy, minz:maxz]
        image2 = image2[minx:maxx, miny:maxy, minz:maxz]
        label = label1[minx:maxx, miny:maxy, minz:maxz]
        image1 = (image1 - np.mean(image1)) / np.std(image1)
        image2 = (image2 - np.mean(image2)) / np.std(image2)
        image1 = image1.astype(np.float32)
        image2 = image2.astype(np.float32)

        f = h5py.File(item.replace('.nii.gz', '_norm.h5').replace("/ASOCA_dataset/mask/", "/ASOCA_DWTLH/"), 'w')
        f.create_dataset('image1', data=image1, compression="gzip")
        f.create_dataset('image2', data=image2, compression="gzip")
        f.create_dataset('label', data=label, compression="gzip")
        print(f['image1'][:].shape) 
        print(f['image2'][:].shape)
        print(f['label'][:].shape)
        f.close()
if __name__ == '__main__':
    covert_h5()