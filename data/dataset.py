import os

import cv2
import torch
from PIL import Image
import random
import numpy as np


def merge_classes(target, n_class, n_tasks):
    assert n_class >= 2 * n_tasks
    nc_per_task = n_class // (2 * n_tasks)
    target = list(np.array(target) // nc_per_task)
    return target




class CUB200(torch.utils.data.Dataset):
    def __init__(self, data_dir, split, transform, target_transform, n_tasks=20):
        self.data_dir = data_dir
        self.split = split # train or test
        self.transform = transform
        self.target_transform = target_transform
        CUB_TRAIN_LIST = 'dataset_lists/CUB_train_list.txt'
        CUB_TEST_LIST = 'dataset_lists/CUB_test_list.txt'
        if split == 'train':
            file_name = CUB_TRAIN_LIST
        elif split == 'test':
            file_name = CUB_TEST_LIST
        else:
            raise 'error'
        self.img_paths = []
        self.targets = []
        with open(file_name) as f:
            for line in f:
                img_name, img_label = line.split()
                img_path = data_dir.rstrip('\/') + '/' + img_name
                self.img_paths.append(img_path)
                self.targets.append(int(img_label))

        self.targets = merge_classes(self.targets, n_class=200, n_tasks=n_tasks)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        img_path, target = self.img_paths[index], self.targets[index]

        img = Image.open(img_path).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


class AWA2(torch.utils.data.Dataset):
    def __init__(self, data_dir, split, transform, target_transform):
        self.data_dir = data_dir
        self.split = split # train or test
        self.transform = transform
        self.target_transform = target_transform
        AWA_TRAIN_LIST = 'dataset_lists/AWA_train_list.txt'
        AWA_TEST_LIST = 'dataset_lists/AWA_test_list.txt'
        if split == 'train':
            file_name = AWA_TRAIN_LIST
        elif split == 'test':
            file_name = AWA_TEST_LIST
        else:
            raise 'error'
        self.img_paths = []
        self.targets = []
        with open(file_name) as f:
            for line in f:
                img_name, img_label = line.split()
                img_path = data_dir.rstrip('\/') + '/' + img_name
                self.img_paths.append(img_path)
                self.targets.append(int(img_label))

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        img_path, target = self.img_paths[index], self.targets[index]

        img = Image.open(img_path).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target