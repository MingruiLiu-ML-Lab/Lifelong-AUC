import torch, torchvision
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import random, copy
import argparse
import numpy as np
from torchvision import transforms
from .dataset import *


_imagenet_transfoms = transforms.Compose([transforms.Resize([224, 224]),
                               transforms.ToTensor(),
                               transforms.Normalize((0.6678, 0.5298, 0.5245), (0.1333, 0.1476, 0.1590))
                               ])
_small_transfoms = transforms.Compose([transforms.Resize([32, 32]),
                               transforms.ToTensor(),
                               transforms.Normalize((0.6678, 0.5298, 0.5245), (0.1333, 0.1476, 0.1590))
                               ])

input_size_dict = {'tiny_imagenet': 32,
                   'SVHN': 32,
                   'CIFAR10': 32,
                   'CUB200': 224,
                   'AWA2': 224,
                   'Stanford_dogs':224,
}


class VisionDataset(object):
    def __init__(self, args):
        self.args = args
        self.supervised_trainloader = get_loader(self.args, indices=None, transforms=_imagenet_transfoms, train=True)
        self.supervised_testloader = get_loader(self.args, indices=None, transforms=_imagenet_transfoms, train=False)
        if args.dataset == 'SVHN':
            self.train_class_labels_dict, self.test_class_labels_dict = classwise_split(
                targets=self.supervised_trainloader.dataset.labels), classwise_split(
                targets=self.supervised_testloader.dataset.labels)
        else:
            self.train_class_labels_dict, self.test_class_labels_dict = classwise_split(
                targets=self.supervised_trainloader.dataset.targets), classwise_split(
                targets=self.supervised_testloader.dataset.targets)
        self.num_classes = len(self.train_class_labels_dict)
        self.num_tasks = self.num_classes // 2
        cl_class_list = list(range(self.num_classes))
        random.shuffle(cl_class_list)
        continual_target_transform = ReorderTargets(cl_class_list)

        self.train_task_loaders = []
        self.test_task_loaders = []
        self.n_input = input_size_dict[args.dataset]

        # task loader
        for i in range(self.num_tasks):
            trainidx = []
            testidx = []
            neg_cls, pos_cls = cl_class_list[2*i], cl_class_list[2*i + 1]
            # make imbalanced
            if args.make_imbalanced == 'yes':
                keep_num_pos = int((args.imratio / (1 - args.imratio)) * len(self.train_class_labels_dict[neg_cls]))
                self.train_class_labels_dict[pos_cls] = self.train_class_labels_dict[pos_cls][:keep_num_pos]

            trainidx += self.train_class_labels_dict[pos_cls] + self.train_class_labels_dict[neg_cls]
            testidx += self.test_class_labels_dict[pos_cls] + self.test_class_labels_dict[neg_cls]

            trainidx += trainidx

            train_loader = get_loader(args, indices=trainidx, transforms=_imagenet_transfoms, train=True,
                                      target_transforms=continual_target_transform)
            test_loader = get_loader(args, indices=testidx, transforms=_imagenet_transfoms, train=False,
                                      target_transforms=continual_target_transform)
            self.train_task_loaders.append(train_loader)
            self.test_task_loaders.append(test_loader)

        # balanced  for Gdumb
        mem_per_cls = args.n_memories // 2
        trainidx = []
        for cl in cl_class_list:  # Selects classes from the continual learning list and loads memory and test indices, which are then passed to a subset sampler
            num_memory_samples = min(len(self.train_class_labels_dict[cl][:]), mem_per_cls)
            trainidx += self.train_class_labels_dict[cl][
                        :num_memory_samples]  # This is class-balanced greedy sampling (Selects the first n samples).
        self.memory_loader = get_loader(args, indices=trainidx, transforms=_imagenet_transfoms, train=True,
                                      target_transforms=continual_target_transform)


def get_loader(args, indices, transforms, train, shuffle=True, target_transforms=None):
    sampler = None
    if indices is not None: sampler = SubsetRandomSampler(indices) if (shuffle and train) else SubsetSequentialSampler(
        indices)

    if args.dataset == 'CUB200':
        split = 'train' if train else 'test'
        dataset = CUB200(data_dir=args.data_path, split=split, transform=transforms,
                         target_transform=target_transforms)
        return DataLoader(dataset, sampler=sampler, num_workers=2, batch_size=args.batch_size)
    elif args.dataset == 'AWA2':
        split = 'train' if train else 'test'
        dataset = AWA2(data_dir=args.data_path, split=split, transform=transforms,
                       target_transform=target_transforms)
        return DataLoader(dataset, sampler=sampler, num_workers=2, batch_size=args.batch_size)
    else:
        raise 'error'


def classwise_split(targets):
    """
    Returns a dictionary with classwise indices for any class key given labels array.
    Arguments:
        indices (sequence): a sequence of indices
    """
    targets = np.array(targets)
    indices = targets.argsort()
    class_labels_dict = dict()

    for idx in indices:
        if targets[idx] in class_labels_dict: class_labels_dict[targets[idx]].append(idx)
        else: class_labels_dict[targets[idx]] = [idx]

    return class_labels_dict


class SubsetSequentialSampler(torch.utils.data.Sampler):
    """
    Samples elements sequentially from a given list of indices, without replacement.
    Arguments:
        indices (sequence): a sequence of indices
    """

    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return (self.indices[i] for i in range(len(self.indices)))

    def __len__(self):
        return len(self.indices)


class ReorderTargets(object):
    """
    Converts the class-orders to 0 -- (n-1) irrespective of order passed.
    """
    def __init__(self, class_order):
        self.class_order = np.array(class_order)

    def __call__(self, target):
        return np.where(self.class_order==target)[0][0]
