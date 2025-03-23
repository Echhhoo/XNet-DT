import nibabel as nib
import numpy as np
import os
import re
import matplotlib.pyplot as plt

def slice_nii_image(input_path, output_root):
    img = nib.load(input_path)
    data = img.get_fdata()
    # 提取文件名部分，不包括扩展名
    base_name = os.path.basename(input_path).split('.')[0]
    folder_path = os.path.dirname(input_path)
    # 获取相对于输出根目录的相对文件夹路径
    relative_folder_path = os.path.relpath(folder_path, start=output_root)
    output_folder = os.path.join(output_root, relative_folder_path)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    for slice_idx in range(0, data.shape[2], 5):
        slice_2d = data[:, :, slice_idx]
        # 使用正则表达式去除非法字符，生成正规的文件名
        clean_name = re.sub(r'[\\/*?:"<>|]', '', base_name)
        print("hello")
        output_filename = os.path.join(output_folder, f'{clean_name}_slice_{slice_idx:03d}.png')
        plt.imsave(output_filename, slice_2d, cmap='gray')

def process_folder(folder_path, output_root):
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('_1.nii.gz'):
                input_path = os.path.join(root, file)
                print(input_path)
                slice_nii_image(input_path, output_root)

input_folder = '/home/ac/datb/wch/ASOCA/'
output_folder = '/home/ac/datb/wch/ASOCA2d/'
process_folder(input_folder, output_folder)