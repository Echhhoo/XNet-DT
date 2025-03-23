import os
import shutil

parent_folder = '/home/ac/datb/wch/imagecasss'
pa_folder='/home/ac/datb/wch/imagecass'
img_folder = os.path.join(pa_folder,'img')
seg_folder = os.path.join(pa_folder, 'mask')

if not os.path.exists(img_folder):
    os.mkdir(img_folder)
if not os.path.exists(seg_folder):
    os.mkdir(seg_folder)

for folder_name in os.listdir(parent_folder):
    if os.path.isdir(os.path.join(parent_folder, folder_name)):
        img_file_path = os.path.join(parent_folder, folder_name, 'img.nii.gz')
        seg_file_path = os.path.join(parent_folder, folder_name, 'label.nii.gz')
        new_img_file_name = f'{folder_name}.nii.gz'
        new_seg_file_name = f'{folder_name}.nii.gz'
        shutil.move(img_file_path, os.path.join(img_folder, new_img_file_name))
        shutil.move(seg_file_path, os.path.join(seg_folder, new_seg_file_name))