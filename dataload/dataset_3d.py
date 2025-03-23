import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import cv2
import numpy as np
import torchio as tio
import SimpleITK as sitk
from torchio.data import UniformSampler, LabelSampler
import torch.nn.functional as F

class dataset_it(Dataset):
    def __init__(self, data_dir, input1, transform_1, queue_length=20, samples_per_volume=5, patch_size=128, num_workers=8, shuffle_subjects=True, shuffle_patches=True, sup=True, num_images=None):
        super(dataset_it, self).__init__()

        self.subjects_1 = []

        image_dir_1 = data_dir + '/' + input1
        if sup:
            mask_dir = data_dir + '/mask'

        for i in os.listdir(image_dir_1):
            image_path_1 = os.path.join(image_dir_1, i)
            if sup:
                mask_path = os.path.join(mask_dir, i)
                subject_1 = tio.Subject(image=tio.ScalarImage(image_path_1), mask=tio.LabelMap(mask_path), ID=i)
            else:
                subject_1 = tio.Subject(image=tio.ScalarImage(image_path_1), ID=i)

            self.subjects_1.append(subject_1)

        if num_images is not None:
            len_img_paths = len(self.subjects_1)
            quotient = num_images // len_img_paths
            remainder = num_images % len_img_paths

            if num_images <= len_img_paths:
                self.subjects_1 = self.subjects_1[:num_images]
            else:
                rand_indices = torch.randperm(len_img_paths).tolist()
                new_indices = rand_indices[:remainder]

                self.subjects_1 = self.subjects_1 * quotient
                self.subjects_1 += [self.subjects_1[i] for i in new_indices]

        self.dataset_1 = tio.SubjectsDataset(self.subjects_1, transform=transform_1)

        self.queue_train_set_1 = tio.Queue(
            subjects_dataset=self.dataset_1,
            max_length=queue_length,
            samples_per_volume=samples_per_volume,
            sampler=UniformSampler(patch_size),
            # sampler=LabelSampler(patch_size),
            num_workers=num_workers,
            shuffle_subjects=shuffle_subjects,
            shuffle_patches=shuffle_patches
        )
        
    def __getitem__(self, index):
        subject=self.subjects_1[index]
        image=np.array(subject['image'])

        
        ID=subject['ID']
        
        # 下采样因子
        downsample_factor = 4

        # 进行下采样
        image = image[::downsample_factor, ::downsample_factor, ::downsample_factor]
        
        original_shape = image.shape
        center_z = original_shape[3] // 2
        start = center_z - 64  # to get 128 in total
        image = image[:, :, :, start:start + 128]
       
        mask=np.array(subject['mask'])
        mask = mask[::downsample_factor, ::downsample_factor, ::downsample_factor]
        mask = mask[:, :, :, start:start + 128]
        sample = {'image': image,'mask': mask, 'ID': ID}    

        return sample
    
    

    def pad_tensor(self, tensor, target_length):
        current_length = tensor.shape[0]
        padding_length = target_length - current_length
        padding = ((0, padding_length), *(tuple(0 for _ in range(tensor.ndim - 1))))
        return F.pad(torch.from_numpy(tensor), padding, 'constant', 0).numpy()
    
    def __len__(self):
        return len(self.subjects_1)

        
     
    


class dataset_it_dtc(Dataset):
    def __init__(self, data_dir, input1, num_classes, transform_1, queue_length=20, samples_per_volume=5, patch_size=128, num_workers=8, shuffle_subjects=True, shuffle_patches=True, sup=True, num_images=None):
        super(dataset_it_dtc, self).__init__()

        self.subjects_1 = []

        image_dir_1 = data_dir + '/' + input1
        if sup:
            mask_dir_1 = data_dir + '/mask'
            mask_dir_2 = data_dir + '/mask_sdf1'
            if num_classes == 3:
                mask_dir_3 = data_dir + '/mask_sdf2'

        for i in os.listdir(image_dir_1):
            image_path_1 = os.path.join(image_dir_1, i)
            if sup:
                mask_path_1 = os.path.join(mask_dir_1, i)
                mask_path_2 = os.path.join(mask_dir_2, i)
                if num_classes == 3:
                    mask_path_3 = os.path.join(mask_dir_3, i)
                    subject_1 = tio.Subject(
                        image=tio.ScalarImage(image_path_1),
                        mask=tio.LabelMap(mask_path_1),
                        mask2=tio.LabelMap(mask_path_2),
                        mask3=tio.LabelMap(mask_path_3),
                        ID=i)
                else:
                    subject_1 = tio.Subject(
                        image=tio.ScalarImage(image_path_1),
                        mask=tio.LabelMap(mask_path_1),
                        mask2=tio.LabelMap(mask_path_2),
                        ID=i)
            else:
                subject_1 = tio.Subject(image=tio.ScalarImage(image_path_1), ID=i)

            self.subjects_1.append(subject_1)

        if num_images is not None:
            len_img_paths = len(self.subjects_1)
            quotient = num_images // len_img_paths
            remainder = num_images % len_img_paths

            if num_images <= len_img_paths:
                self.subjects_1 = self.subjects_1[:num_images]
            else:
                rand_indices = torch.randperm(len_img_paths).tolist()
                new_indices = rand_indices[:remainder]

                self.subjects_1 = self.subjects_1 * quotient
                self.subjects_1 += [self.subjects_1[i] for i in new_indices]

        self.dataset_1 = tio.SubjectsDataset(self.subjects_1, transform=transform_1)

        self.queue_train_set_1 = tio.Queue(
            subjects_dataset=self.dataset_1,
            max_length=queue_length,
            samples_per_volume=samples_per_volume,
            sampler=UniformSampler(patch_size),
            # sampler=LabelSampler(patch_size),
            num_workers=num_workers,
            shuffle_subjects=shuffle_subjects,
            shuffle_patches=shuffle_patches
        )

class dataset_iit(Dataset):
    def __init__(self, data_dir, input1, input2, transform_1, queue_length=20, samples_per_volume=5, patch_size=128, num_workers=8, shuffle_subjects=True, shuffle_patches=True, sup=True, num_images=None):
        super(dataset_iit, self).__init__()
        img_paths_1 = []
        img_paths_2 = []
        mask_paths = []

        self.subjects_1 = []

        image_dir_1 = data_dir + '/' + input1
        image_dir_2 = data_dir + '/' + input2

        if sup:
            mask_dir_1 = data_dir + '/mask'

        for i in os.listdir(image_dir_1):
            image_path_1 = os.path.join(image_dir_1, i)
            img_paths_1.append(image_path_1)
            
            image_path_2 = os.path.join(image_dir_2, i)
            img_paths_2.append(image_path_2)
            
            if sup:
                mask_path_1 = os.path.join(mask_dir_1, i)
                mask_paths.append(mask_path_1)
                
                subject_1 = tio.Subject(image=tio.ScalarImage(image_path_1), image2=tio.ScalarImage(image_path_2), mask=tio.LabelMap(mask_path_1), ID=i)
            else:
                subject_1 = tio.Subject(image=tio.ScalarImage(image_path_1), image2=tio.ScalarImage(image_path_2), ID=i)
               
            self.subjects_1.append(subject_1)

        if num_images is not None:
            len_img_paths = len(self.subjects_1)
            quotient = num_images // len_img_paths
            remainder = num_images % len_img_paths

            if num_images <= len_img_paths:
                self.subjects_1 = self.subjects_1[:num_images]
            else:
                rand_indices = torch.randperm(len_img_paths).tolist()
                new_indices = rand_indices[:remainder]

                self.subjects_1 = self.subjects_1 * quotient
                self.subjects_1 += [self.subjects_1[i] for i in new_indices]

        self.dataset_1 = tio.SubjectsDataset(self.subjects_1, transform=transform_1)

        self.queue_train_set_1 = tio.Queue(
            subjects_dataset=self.dataset_1,
            max_length=queue_length,
            samples_per_volume=samples_per_volume,
            sampler=UniformSampler(patch_size),
            # sampler=LabelSampler(patch_size),
            num_workers=num_workers,
            shuffle_subjects=shuffle_subjects,
            shuffle_patches=shuffle_patches
        )
        self.img_paths_1 = img_paths_1
        self.img_paths_2 = img_paths_2
        self.mask_paths = mask_paths
        self.transform_1=transform_1
        self.sup = sup
        
    def __getitem__(self, index):
        subject=self.subjects_1[index]
        image_1=np.array(subject['image'])
        image_2=np.array(subject['image2'])
        
        ID=subject['ID']
        
        # 下采样因子
        downsample_factor = 2

        # 进行下采样
        image_1 = image_1[::downsample_factor, ::downsample_factor, ::downsample_factor]
        image_2 = image_2[::downsample_factor, ::downsample_factor, ::downsample_factor]
        
        original_shape = image_1.shape
        center_z = original_shape[3] // 2
        start = center_z - 64  # to get 128 in total
        image_1 = image_1[:, :, :, start:start + 128]
        image_2 = image_2[:, :, :, start:start + 128]

       
        if self.sup:
            mask=np.array(subject['mask'])
            mask = mask[::downsample_factor, ::downsample_factor, ::downsample_factor]
            mask = mask[:, :, :, start:start + 128]
            sample = {'image': image_1, 'image_2': image_2,'mask': mask, 'ID': ID}    
        else:
            sample = {'image': image_1, 'image_2': image_2, 'ID': ID}
        return sample
    
    

    def pad_tensor(self, tensor, target_length):
        current_length = tensor.shape[0]
        padding_length = target_length - current_length
        padding = ((0, padding_length), *(tuple(0 for _ in range(tensor.ndim - 1))))
        return F.pad(torch.from_numpy(tensor), padding, 'constant', 0).numpy()
    
    def __len__(self):
        return len(self.subjects_1)


class dataset_iit_conresnet(Dataset):
    def __init__(self, data_dir, input1, input2, transform_1, queue_length=20, samples_per_volume=5, patch_size=128, num_workers=8, shuffle_subjects=True, shuffle_patches=True, sup=True, num_images=None):
        super(dataset_iit_conresnet, self).__init__()

        self.subjects_1 = []

        image_dir_1 = data_dir + '/' + input1
        image_dir_2 = data_dir + '/' + input2

        if sup:
            mask_dir_1 = data_dir + '/mask'
            mask_dir_2 = data_dir + '/mask_res'

        for i in os.listdir(image_dir_1):
            image_path_1 = os.path.join(image_dir_1, i)
            image_path_2 = os.path.join(image_dir_2, i)
            if sup:
                mask_path_1 = os.path.join(mask_dir_1, i)
                mask_path_2 = os.path.join(mask_dir_2, i)
                subject_1 = tio.Subject(image=tio.ScalarImage(image_path_1), image2=tio.ScalarImage(image_path_2), mask=tio.LabelMap(mask_path_1), mask2=tio.LabelMap(mask_path_2), ID=i)
            else:
                subject_1 = tio.Subject(image=tio.ScalarImage(image_path_1), image2=tio.ScalarImage(image_path_2), ID=i)

            self.subjects_1.append(subject_1)

        if num_images is not None:
            len_img_paths = len(self.subjects_1)
            quotient = num_images // len_img_paths
            remainder = num_images % len_img_paths

            if num_images <= len_img_paths:
                self.subjects_1 = self.subjects_1[:num_images]
            else:
                rand_indices = torch.randperm(len_img_paths).tolist()
                new_indices = rand_indices[:remainder]

                self.subjects_1 = self.subjects_1 * quotient
                self.subjects_1 += [self.subjects_1[i] for i in new_indices]

        self.dataset_1 = tio.SubjectsDataset(self.subjects_1, transform=transform_1)

        self.queue_train_set_1 = tio.Queue(
            subjects_dataset=self.dataset_1,
            max_length=queue_length,
            samples_per_volume=samples_per_volume,
            sampler=UniformSampler(patch_size),
            # sampler=LabelSampler(patch_size),
            num_workers=num_workers,
            shuffle_subjects=shuffle_subjects,
            shuffle_patches=shuffle_patches
        )

