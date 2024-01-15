import torch
import torchvision
import torchvision.transforms as transforms
from conf import settings
import random
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os


def cifar10_loader(args):
    # data preprocessing:
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    trainset = torchvision.datasets.CIFAR10(root=settings.DATA_PATH, train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root=settings.DATA_PATH, train=False, download=True, transform=transform_test)

    return trainset, testset
    

def cifar100_loader(args):
    transform_train = transforms.Compose([
    transforms.Pad(4, padding_mode='reflect'),
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32),
    transforms.ToTensor(),
    transforms.Normalize(np.array([125.3, 123.0, 113.9]) / 255.0,
                        np.array([63.0, 62.1, 66.7]) / 255.0)
    ])

    transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(np.array([125.3, 123.0, 113.9]) / 255.0,
                        np.array([63.0, 62.1, 66.7]) / 255.0),
    ])
    trainset = torchvision.datasets.CIFAR100(root=settings.DATA_PATH, train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR100(root=settings.DATA_PATH, train=False, download=True, transform=transform_test)

    return trainset, testset


class Cutout(object):
    """Randomly mask out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """

    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h = img.size(2)
        w = img.size(3)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)
        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)
        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask
        return img


class NCaltech101(Dataset):
    def __init__(self, data_path='datapath/frames_number_6_split_by_time',
                 data_type='train', transform=True):

        self.filepath = os.path.join(data_path)
        self.clslist = os.listdir(self.filepath)
        self.clslist.sort()

        self.dvs_filelist = []
        self.targets = []
        self.resize = transforms.Resize(size=(48, 48), interpolation=torchvision.transforms.InterpolationMode.NEAREST)

        for i, cls in enumerate(self.clslist):
            # print (i, cls)
            file_list = os.listdir(os.path.join(self.filepath, cls))
            num_file = len(file_list)

            cut_idx = int(num_file * 0.9)
            train_file_list = file_list[:cut_idx]
            test_split_list = file_list[cut_idx:]
            for file in file_list:
                if data_type == 'train':
                    if file in train_file_list:
                        self.dvs_filelist.append(os.path.join(self.filepath, cls, file))
                        self.targets.append(i)
                else:
                    if file in test_split_list:
                        self.dvs_filelist.append(os.path.join(self.filepath, cls, file))
                        self.targets.append(i)

        self.data_num = len(self.dvs_filelist)
        self.data_type = data_type
        if data_type != 'train':
            counts = np.unique(np.array(self.targets), return_counts=True)[1]
            class_weights = counts.sum() / (counts * len(counts))
            self.class_weights = torch.Tensor(class_weights)
        self.classes = range(101)
        self.transform = transform
        self.rotate = transforms.RandomRotation(degrees=15)
        self.shearx = transforms.RandomAffine(degrees=0, shear=(-15, 15))

    def __getitem__(self, index):
        file_pth = self.dvs_filelist[index]
        label = self.targets[index]
        data = torch.from_numpy(np.load(file_pth)['frames']).float()
        data = self.resize(data)

        if self.transform:

            choices = ['roll', 'rotate', 'shear']
            aug = np.random.choice(choices)
            if aug == 'roll':
                off1 = random.randint(-3, 3)
                off2 = random.randint(-3, 3)
                data = torch.roll(data, shifts=(off1, off2), dims=(2, 3))
            if aug == 'rotate':
                data = self.rotate(data)
            if aug == 'shear':
                data = self.shearx(data)

        return data, label

    def __len__(self):
        return self.data_num


def build_ncaltech(transform=False):
    train_dataset = NCaltech101(transform=transform)
    val_dataset = NCaltech101(data_type='test', transform=False)

    return train_dataset, val_dataset