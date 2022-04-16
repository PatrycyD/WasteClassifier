import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import os
import WasteClassifier.config as config


class DataManager:

    def __init__(self, data_path, transform_type: str = 'test', batch_size: int = 10):
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_of_classes = None

        if transform_type == 'test':
            self.transform = transforms.Compose([
                                                transforms.CenterCrop(224),
                                                transforms.ToTensor(),
                                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                                ])
        else:
            self.transform = transforms.Compose([
                                                transforms.RandomRotation(10),
                                                transforms.RandomHorizontalFlip(),
                                                transforms.CenterCrop(224),
                                                transforms.ToTensor(),
                                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                                ])

    def return_dataset_and_loader(self, manual_seed: int = 42, return_loader: bool = True, shuffle: bool = True):

        data = datasets.ImageFolder(self.data_path, transform=self.transform)

        torch.manual_seed(manual_seed)
        if return_loader:
            loader = DataLoader(data, batch_size=self.batch_size, shuffle=shuffle)
            return loader, data

        return data

    def return_dataset_and_laoder_of_n_photos(self, n: int, display_photos: bool = True, shuffle: bool = True):

        data = datasets.ImageFolder(self.data_path, transform=self.transform)
        data = torch.utils.data.Subset(data, np.random.choice(len(data), n))

        loader = DataLoader(data, self.batch_size, shuffle=shuffle)

        return data, loader

    def get_number_of_classes(self):
        num_of_classes = len(next(os.walk(self.data_path))[1])
        self.num_of_classes = num_of_classes

        return num_of_classes


if __name__ == '__main__':
    read_to_loader_n_photos('/home/peprycy/WasteClassifier/Data/TrashNet', 5)
