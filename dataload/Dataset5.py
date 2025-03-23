import itertools
import os
import random
import re
from glob import glob
import math
import cv2
import h5py
import numpy as np
from medpy import metric
from tqdm import tqdm
import torch
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from skimage import exposure
from sklearn.model_selection import KFold, train_test_split
from torch.utils.data import Dataset
import copy
from torch.utils.data.sampler import Sampler
try:  # SciPy >= 0.19
    from scipy.special import comb
except ImportError:
    from scipy.misc import comb

class BaseDataSets(Dataset):
    def __init__(self, base_dir=None, save_dir=None,  labeled_type="labeled", labeled_ratio=10,transform=None,split='train',fold=0,cross_val=True):
        
        self._base_dir = base_dir
        self.sample_list = []
        self.split = split
        self.transform = transform
        self.labeled_type = labeled_type
        self.all_volumes = sorted(os.listdir(self._base_dir + "/"))
        if cross_val:  # 5-fold  cross validation
            train_ids, val_ids = self._get_fold_ids(fold)
        else:
            train_ids, val_ids, _ = self._get_split_ids()
        train_ids = sorted(train_ids)
        all_labeled_ids = train_ids[::labeled_ratio]
        train_labeled_path = save_dir + 'train_labeled.list'
        train_unlabeled_path = save_dir + 'train_unlabeled.list'
        val_path = save_dir + 'val.list'
        
        if self.split == 'train':
            self.all_volumes = os.listdir(self._base_dir + "/")
            self.sample_list = []
            labeled_ids = [i for i in all_labeled_ids if i in train_ids]
            unlabeled_ids = [i for i in train_ids if i not in labeled_ids]
            if self.labeled_type == "labeled":
                # print("Labeled IDs", labeled_ids)
                with open(train_labeled_path, 'w+') as f:
                    for ids in labeled_ids:
                        new_data_list = list(filter(lambda x: re.match('{}.*'.format(ids.replace(".h5", "")), x) != None, self.all_volumes))
                        f.write(str(new_data_list)+'\n')
                        self.sample_list.extend(new_data_list)
                print("total labeled {} samples".format(len(self.sample_list)))
            else:
                # print("Unlabeled IDs", unlabeled_ids)
                with open(train_unlabeled_path, 'w+') as f:
                    for ids in unlabeled_ids:
                        new_data_list = list(filter(lambda x: re.match('{}.*'.format(ids.replace(".h5", "")), x) != None, self.all_volumes))
                        f.write(str(new_data_list)+'\n')
                        self.sample_list.extend(new_data_list)
                print("total unlabeled {} samples".format(len(self.sample_list)))
        elif self.split == 'val':
            print("val_ids", val_ids)
            self.sample_list = []
            with open(val_path, 'w+') as f:
                for ids in val_ids:
                    new_data_list = list(filter(lambda x: re.match('{}.*'.format(ids.replace(".h5", "")), x) != None, self.all_volumes))
                    f.write(str(new_data_list) + '\n')
                    self.sample_list.extend(new_data_list)
            print("total val {} samples".format(len(self.sample_list)))
    def _get_fold_ids(self, fold):
        folds = KFold(n_splits=4, shuffle=True, random_state=1)
        all_cases = np.array(self.all_volumes)
        k_fold_data = []
        for trn_idx, val_idx in folds.split(all_cases):
            k_fold_data.append([all_cases[trn_idx], all_cases[val_idx]])
        # print((np.array(k_fold_data)).shape)

        train_set = k_fold_data[fold][0]
        # print((np.array(train_set)).shape)
        test_set = k_fold_data[fold][1]
        # print((np.array(test_set)).shape)

        return train_set, test_set

    def _get_split_ids(self):
        all_cases = np.array(self.all_volumes)
        rest_set, test_set = train_test_split(all_cases, test_size=int(
            len(self.all_volumes)*0.2), shuffle=True, random_state=1234)
        train_set, val_set = train_test_split(rest_set, test_size=int(
            len(self.all_volumes)*0.1), shuffle=True, random_state=1234)
        print("test_set", sorted(test_set))
        return train_set, val_set, test_set

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        case = self.sample_list[idx]
        if self.split == "train":
            h5f = h5py.File(self._base_dir +
                            "/{}".format(case), 'r')
        else:
            h5f = h5py.File(self._base_dir +
                            "/{}".format(case), 'r')
        if self.split == "train":
            image1 = h5f['image1'][:]
            image2 = h5f['image2'][:]
            image3 = h5f['image3'][:]
            image4 = h5f['image4'][:]
            label = h5f["label"][:]
            sample = {'image1': image1,'image2': image2, 'image3': image3,'image4': image4,'label': label}
            sample = self.transform(sample)
        else:
            image1 = h5f['image1'][:]
            image2 = h5f['image2'][:]
            image3 = h5f['image3'][:]
            image4 = h5f['image4'][:]
            label = h5f['label'][:].astype(np.int16)
            sample = {'image1': image1,'image2': image2,'image3': image3, 'image4': image4,'label': label}
            sample = self.transform(sample)
        sample["idx"] = case.split("_")[0]
        return sample


def bernstein_poly(i, n, t):
    """
     The Bernstein polynomial of n, i as a function of t
    """
    return comb(n, i) * (t**(n-i)) * (1 - t)**i


def bezier_curve(points, nTimes=1000):
    """
       Given a set of control points, return the
       bezier curve defined by the control points.
       Control points should be a list of lists, or list of tuples
       such as [ [1,1],
                 [2,3],
                 [4,5], ..[Xn, Yn] ]
        nTimes is the number of time steps, defaults to 1000
        See http://processingjs.nihongoresources.com/bezierinfo/
    """

    nPoints = len(points)
    wPoints = np.array([p[0] for p in points])
    hPoints = np.array([p[1] for p in points])
    dPoints = np.array([p[2] for p in points])
    t = np.linspace(0.0, 1.0, nTimes)

    polynomial_array = np.array(
        [bernstein_poly(i, nPoints-1, t) for i in range(0, nPoints)])

    wvals = np.dot(wPoints, polynomial_array)
    hvals = np.dot(hPoints, polynomial_array)
    dvals = np.dot(dPoints, polynomial_array)

    return wvals, hvals, dvals


def nonlinear_transformation(w, label, prob=0.5):
    if random.random() >= prob:
        return w, label
    points = [[0, 0, 0], [random.random(), random.random(),random.random()], [
        random.random(), random.random(),random.random()], [1, 1, 1]]
    wPoints = np.array([p[0] for p in points])
    hPoints = np.array([p[1] for p in points])
    dPoints = np.array([p[2] for p in points])
    wvals, hvals, dvals = bezier_curve(points, nTimes=100000)
    if random.random() < 0.5:
        # Half change to get flip
        wvals = np.sort(wvals)
    else:
        wvals, hvals, dvals = np.sort(wvals), np.sort(hvals),np.sort(dvals)
    nonlinear_w = np.interp(w, wvals, hvals, dvals)
    return nonlinear_w, label


def random_rescale_intensity(image1,image2,image3,image4, label):
    image1 = exposure.rescale_intensity(image1)
    image2 = exposure.rescale_intensity(image2)
    image3 = exposure.rescale_intensity(image3)
    image4 = exposure.rescale_intensity(image4)
    return image1,image2,image3,image4,label


def random_equalize_hist(image1,image2,image3,image4,label):
    image1 = exposure.equalize_hist(image1)
    image2 = exposure.equalize_hist(image2)
    image3 = exposure.equalize_hist(image3)
    image4 = exposure.equalize_hist(image4)
    return image1,image2,image3,image4,label


class RandomGenerator_Strong_Weak(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image1,image2,image3,image4,label = sample['image1'], sample['image2'],sample['image3'],sample['image4'],sample['label']
        if random.random() > 0.5:
            image1,image2,image3,image4,label = RandomRotFlip(image1,image2,image3,image4, label)
        if random.random() > 0.5:
            image1,image2,image3,image4,label = RandomRotFlip(image1,image2,image3,image4, label, cval=0)
        if random.random() > 0.5:
            image1,image2,image3,image4,label = RandomNoise(image1,image2,image3,image4, label)

        w, h, d = image1.shape
        image_w1 = copy.deepcopy(image1)
        image_w1 = zoom(
            image_w1, (self.output_size[0] / w, self.output_size[1] / h, self.output_size[2] / d), order=0)
        
        image_w2 = copy.deepcopy(image2)
        image_w2 = zoom(
            image_w2, (self.output_size[0] / w, self.output_size[1] / h, self.output_size[2] / d), order=0)
        
        image_w3 = copy.deepcopy(image3)
        image_w3 = zoom(
            image_w3, (self.output_size[0] / w, self.output_size[1] / h, self.output_size[2] / d), order=0)
        
        image_w4 = copy.deepcopy(image4)
        image_w4 = zoom(
            image_w4, (self.output_size[0] / w, self.output_size[1] / h, self.output_size[2] / d), order=0)

        if random.random() > 0.33:
            image1,image2,image3,image4, label = nonlinear_transformation(image1,image2,image3,image4,label)
        elif random.random() < 0.66 and random.random() > 0.33:
            image1,image2,image3,image4, label = random_rescale_intensity(image1,image2,image3,image4, label)
        else:
            image1,image2,image3,image4, label = random_equalize_hist(image1,image2,image3,image4, label)
        image_s1 = image1
        image_s1 = zoom(
            image_s1, (self.output_size[0] / w, self.output_size[1] / h, self.output_size[2] / d), order=0)
        
        image_s2 = image2
        image_s2 = zoom(
            image_s2, (self.output_size[0] / w, self.output_size[1] / h, self.output_size[2] / d), order=0)

        image_s3 = image3
        image_s3 = zoom(
            image_s3, (self.output_size[0] / w, self.output_size[1] / h, self.output_size[2] / d), order=0)
        
        image_s4 = image4
        image_s4 = zoom(
            image_s4, (self.output_size[0] / w, self.output_size[1] / h, self.output_size[2] / d), order=0)
        label = zoom(
            label, (self.output_size[0] / w, self.output_size[1] / h, self.output_size[2] / d), order=0)
        
        image_w1 = torch.from_numpy(
            image_w1.astype(np.float32)).unsqueeze(0)
        image_s1 = torch.from_numpy(
            image_s1.astype(np.float32)).unsqueeze(0)
        image_w2 = torch.from_numpy(
            image_w2.astype(np.float32)).unsqueeze(0)
        image_s2 = torch.from_numpy(
            image_s2.astype(np.float32)).unsqueeze(0)
        image_w3 = torch.from_numpy(
            image_w3.astype(np.float32)).unsqueeze(0)
        image_s3 = torch.from_numpy(
            image_s3.astype(np.float32)).unsqueeze(0)
        image_w4 = torch.from_numpy(
            image_w4.astype(np.float32)).unsqueeze(0)
        image_s4 = torch.from_numpy(
            image_s4.astype(np.float32)).unsqueeze(0)
        
        label = torch.from_numpy(label.astype(np.int16))
        sample = {'image_w1': image_w1, 'image_s1': image_s1,'image_w2': image_w2, 'image_s2': image_s2,'image_w3': image_w3, 'image_s3': image_s3,'image_w4': image_w4, 'image_s4': image_s4, 'label': label}
        return sample


class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image1,image2,image3,image4,label = sample['image1'], sample['image2'],sample['image3'],sample['image4'], sample['label']
        if random.random() > 0.5:
            image1,image2,image3,image4,label = RandomRotFlip(image1,image2,image3,image4,label)
        if random.random() <0.2:
            image1,image2,image3,image4,label = RandomRotFlip(image1,image2,image3,image4,label, cval=0)
        if random.random() > 0.9:
            image1,image2,image3,image4,label = RandomNoise(image1,image2,image3,image4,label)
        if random.random() > 0.33:
            image1,image2,image3,image4,label = nonlinear_transformation(image1,image2,image3,image4,label)
        elif random.random() < 0.66 and random.random() > 0.33:
            image1,image2,image3,image4,label = random_rescale_intensity(image1,image2,image3,image4,label)
        elif random.random() > 0.66:
            image1,image2,image3,image4,label = random_equalize_hist(image1,image2,image3,image4,label)
        w, h, d = image1.shape
        image1 = zoom(
            image1, (self.output_size[0] / w, self.output_size[1] / h, self.output_size[2] / d), order=0)
        image2 = zoom(
            image2, (self.output_size[0] / w, self.output_size[1] / h, self.output_size[2] / d), order=0)
        image3 = zoom(
            image3, (self.output_size[0] / w, self.output_size[1] / h, self.output_size[2] / d), order=0)
        image4 = zoom(
            image4, (self.output_size[0] / w, self.output_size[1] / h, self.output_size[2] / d), order=0)
        label = zoom(
            label, (self.output_size[0] / w, self.output_size[1] / h, self.output_size[2] / d), order=0)
        
        image1 = torch.from_numpy(
            image1.astype(np.float32)).unsqueeze(0)
        image2 = torch.from_numpy(
            image2.astype(np.float32)).unsqueeze(0)
        image3 = torch.from_numpy(
            image3.astype(np.float32)).unsqueeze(0)
        image4 = torch.from_numpy(
            image4.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.int16))
        sample = {'image1': image1,'image2': image2,'image3': image3, 'image4': image4,'label': label}
        return sample

class RandomCrop(object):
    """
    Crop randomly the image in a sample
    Args:
    output_size (int): Desired output size
    """

    def __init__(self, output_size, with_sdf=False):
        self.output_size = output_size
        self.with_sdf = with_sdf

    def __call__(self, sample):
        image1, image2,image3,image4,label = sample['image1'], sample['image2'], sample['image3'],sample['image4'],sample['label']
        if self.with_sdf:
            sdf = sample['sdf']

        # pad the sample if necessary
        if label.shape[0] <= self.output_size[0] or label.shape[1] <= self.output_size[1] or label.shape[2] <= \
                self.output_size[2]:
            pw = max((self.output_size[0] - label.shape[0]) // 2 + 3, 0)
            ph = max((self.output_size[1] - label.shape[1]) // 2 + 3, 0)
            pd = max((self.output_size[2] - label.shape[2]) // 2 + 3, 0)
            image1 = np.pad(image1, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
            image2 = np.pad(image2, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
            image3 = np.pad(image3, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
            image4 = np.pad(image4, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
            label = np.pad(label, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
            if self.with_sdf:
                sdf = np.pad(sdf, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)

        (w, h, d) = image1.shape
        # if np.random.uniform() > 0.33:
        #     w1 = np.random.randint((w - self.output_size[0])//4, 3*(w - self.output_size[0])//4)
        #     h1 = np.random.randint((h - self.output_size[1])//4, 3*(h - self.output_size[1])//4)
        # else:
        w1 = np.random.randint(0, w - self.output_size[0])
        h1 = np.random.randint(0, h - self.output_size[1])
        d1 = np.random.randint(0, d - self.output_size[2])

        label = label[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
        image1 = image1[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
        image2 = image2[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
        image3 = image3[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
        image4 = image4[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
        if self.with_sdf:
            sdf = sdf[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
            return {'image1': image1, 'image2': image2,'image3': image3, 'image4': image4, 'label': label, 'sdf': sdf}
        else:
            return {'image1': image1, 'image2': image2,'image3': image3, 'image4': image4, 'label': label}


class RandomRotFlip(object):
    """
    Crop randomly flip the dataset in a sample
    Args:
    output_size (int): Desired output size
    """

    def __call__(self, sample):
        image1,image2, label = sample['image1'], sample['image2'],sample['label']
        k = np.random.randint(0, 4)
        image1 = np.rot90(image1, k)
        image2 = np.rot90(image2, k)
        image3 = np.rot90(image3, k)
        image4 = np.rot90(image4, k)
        label = np.rot90(label, k)
        axis = np.random.randint(0, 2)
        image1 = np.flip(image1, axis=axis).copy()
        image2 = np.flip(image2, axis=axis).copy()
        image3 = np.flip(image3, axis=axis).copy()
        image4 = np.flip(image4, axis=axis).copy()
        label = np.flip(label, axis=axis).copy()

        return {'image1': image1, 'image2': image2,'image3': image3, 'image4': image4,'label': label}


class RandomNoise(object):
    def __init__(self, mu=0, sigma=0.1):
        self.mu = mu
        self.sigma = sigma

    def __call__(self, sample):
        image1,image2,image3,image4,label = sample['image1'], sample['image2'], sample['image3'],sample['image4'],sample['label']
        noise = np.clip(self.sigma * np.random.randn(image1.shape[0], image1.shape[1], image1.shape[2]), -2*self.sigma, 2*self.sigma)
        noise = noise + self.mu
        image1 = image1 + noise
        image2 = image2 + noise
        image3 = image3 + noise
        image4 = image4 + noise
        return {'image1': image1, 'image2': image2,'image3': image3,'image4': image4, 'label': label}


class CreateOnehotLabel(object):
    def __init__(self, num_classes):
        self.num_classes = num_classes

    def __call__(self, sample):
        image1,image2,image3, image4,label = sample['image1'], sample['image2'], sample['image3'],sample['image4'],sample['label']
        onehot_label = np.zeros((self.num_classes, label.shape[0], label.shape[1], label.shape[2]), dtype=np.float32)
        for i in range(self.num_classes):
            onehot_label[i, :, :, :] = (label == i).astype(np.float32)
        return {'image1': image1,'image2': image2,'image3': image3, 'image4': image4,'label': label,'onehot_label':onehot_label}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image1 = sample['image1']
        image2 = sample['image2']
        image3 = sample['image3']
        image4 = sample['image4']
        
        image1 = image1.reshape(1, image1.shape[0], image1.shape[1], image1.shape[2]).astype(np.float32)
        image2 = image2.reshape(1, image2.shape[0], image2.shape[1], image2.shape[2]).astype(np.float32)
        image3 = image3.reshape(1, image3.shape[0], image3.shape[1], image3.shape[2]).astype(np.float32)
        image4 = image4.reshape(1, image4.shape[0], image4.shape[1], image4.shape[2]).astype(np.float32)
        if 'onehot_label' in sample:
            return {'image1': torch.from_numpy(image1),'image2': torch.from_numpy(image2), 'image3': torch.from_numpy(image3),'image4': torch.from_numpy(image4),'label': torch.from_numpy(sample['label']).long(),
                    'onehot_label': torch.from_numpy(sample['onehot_label']).long()}
        else:
            return {'image1': torch.from_numpy(image1),'image2': torch.from_numpy(image2),  'image3': torch.from_numpy(image3),'image4': torch.from_numpy(image4),'label': torch.from_numpy(sample['label']).long()}

class TwoStreamBatchSampler(Sampler):
    """Iterate two sets of indices

    An 'epoch' is one iteration through the primary indices.
    During the epoch, the secondary indices are iterated through
    as many times as needed.
    """
    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size):
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.secondary_batch_size = secondary_batch_size
        self.primary_batch_size = batch_size - secondary_batch_size

        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __iter__(self):
        primary_iter = iterate_once(self.primary_indices)
        secondary_iter = iterate_eternally(self.secondary_indices)
        return (
            primary_batch + secondary_batch
            for (primary_batch, secondary_batch)
            in zip(grouper(primary_iter, self.primary_batch_size),
                    grouper(secondary_iter, self.secondary_batch_size))
        )

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size
def iterate_once(iterable):
    return np.random.permutation(iterable)


def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)
    return itertools.chain.from_iterable(infinite_shuffles())


def grouper(iterable, n):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3) --> ABC DEF"
    args = [iter(iterable)] * n
    return zip(*args)
