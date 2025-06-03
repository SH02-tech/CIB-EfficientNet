import os
import numpy as np
import pandas as pd

from torchvision import datasets, transforms
from torchvision.io import read_image
from torch.utils.data import Dataset
from base import BaseDataLoader


class MnistDataLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        trsfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.data_dir = data_dir
        self.dataset = datasets.MNIST(self.data_dir, train=training, download=True, transform=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


class ISIC2018Dataset(Dataset):
    """
    ISIC2018 data loading demo using BaseDataLoader
    """
    def __init__(self, data_dir, split:str = 'train'):

        # get data, depending on split

        if split == 'train':
            split_data = os.path.join(data_dir + 'ISIC2018_Task3_Training_Input')
            split_labels = os.path.join(data_dir + 'ISIC2018_Task3_Training_GroundTruth/ISIC2018_Task3_Training_GroundTruth.csv')
        elif split == 'val':
            split_data = os.path.join(data_dir + 'ISIC2018_Task3_Validation_Input')
            split_labels = os.path.join(data_dir + 'ISIC2018_Task3_Validation_GroundTruth/ISIC2018_Task3_Validation_GroundTruth.csv')
        elif split == 'test':
            split_data = os.path.join(data_dir + 'ISIC2018_Task3_Test_Input')
            split_labels = os.path.join(data_dir + 'ISIC2018_Task3_Test_GroundTruth/ISIC2018_Task3_Test_GroundTruth.csv')
        else:
            raise ValueError("split must be one of 'train', 'val', or 'test'")
        
        self.img_dir = split_data
        self.img_labels = pd.read_csv(split_labels)

        # get transform (as in https://docs.google.com/presentation/d/1Lup8MnuOkVakDL5-VWx14n-94pEKRq6cY88xoO6MF0w/edit#slide=id.g430d0eaa01_0_300)

        IDENTITY = transforms.Lambda(lambda x: x)

        self.transforms = transforms.Compose([
            transforms.RandomAffine(degrees=360, shear=15) if split == 'train' else IDENTITY,
            transforms.Resize(size=374),
            transforms.CenterCrop(size=374),
            transforms.RandomResizedCrop(size=299, scale=(0.8, 1.0), ratio=(1.0, 1.0)) if split == 'train' else IDENTITY,
            transforms.RandomHorizontalFlip() if split == 'train' else IDENTITY,
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1) if split == 'train' else IDENTITY,
            transforms.Lambda(lambda x: x / 255.0),
            # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def __len__(self):
        return len(self.img_labels)
    
    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, f'{self.img_labels.iloc[idx, 0]}.jpg')
        image = read_image(img_name)
        labels = self.img_labels.iloc[idx, 1:].to_numpy(dtype='int64')
        target_class = np.argmax(labels, axis=0)
        image = self.transforms(image)
        return image, target_class

class ISIC2018DataLoader(BaseDataLoader):
    """
    ISIC2018 data loading demo using BaseDataLoader
    """
    def __init__(self, data_dir, split, batch_size=4, shuffle=True, validation_split=0.0, num_workers=1):
        self.data_dir = data_dir
        self.dataset = ISIC2018Dataset(self.data_dir, split)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

class JacobMedDataset(Dataset):
    """
    JacobMed data loading
    """
    def __init__(self, data_dir, split:str = 'train', reduced_set = False):
        # attributes
        self.data_files = []
        self.data_dict = {}
        self.transforms = None
        self.normalize_mean = [0.5, 0.5, 0.5]
        self.normalize_std = [0.5, 0.5, 0.5]

        with open(os.path.join(data_dir, 'classes.txt'), 'r') as f:
            classes = f.readlines()
            self.data_dict = {x.strip(): idx for idx, x in enumerate(classes)}

        # get data, depending on split
        split_data = os.path.join(data_dir, split)

        # get all data files
        for field in self.data_dict.keys():
            field_dir = os.path.join(split_data, field)
            new_data_files = []
            for file in os.listdir(field_dir):
                if file.endswith('.png'):
                    new_data_files.append(os.path.join(field_dir, file))
            if not reduced_set:
                self.data_files.extend(new_data_files)
            else:
                # add subst of data files
                num_elems = 300 if split == 'train' else 20
                self.data_files.extend(new_data_files[:num_elems])

        self.data_files.sort()

        IDENTITY = transforms.Lambda(lambda x: x)

        self.transforms = transforms.Compose([
            transforms.RandomAffine(degrees=360, shear=15) if split == 'train' else IDENTITY,
            transforms.Resize(size=374),
            transforms.CenterCrop(size=374),
            transforms.RandomResizedCrop(size=299, scale=(0.8, 1.0), ratio=(1.0, 1.0)) if split == 'train' else IDENTITY,
            transforms.RandomHorizontalFlip() if split == 'train' else IDENTITY,
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1) if split == 'train' else IDENTITY,
            transforms.Lambda(lambda x: x / 255.0),
            transforms.Normalize(mean=self.normalize_mean, std=self.normalize_std)
        ])

    def __len__(self):
        return len(self.data_files)
    
    def __getitem__(self, idx):
        img_name = self.data_files[idx]
        image = read_image(img_name)
        label = os.path.basename(os.path.dirname(img_name))
        target_class = self.data_dict[label]
        image = self.transforms(image)
        return image, target_class


class JacobMedDataLoader(BaseDataLoader):
    """
    JacobMed data loading demo using BaseDataLoader
    """
    def __init__(self, data_dir, split, reduced_set=False, batch_size=4, shuffle=True, validation_split=0.0, num_workers=1):
        self.data_dir = data_dir
        self.dataset = JacobMedDataset(self.data_dir, split=split, reduced_set=reduced_set)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)
