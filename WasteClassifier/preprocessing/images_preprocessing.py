import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np


def read_to_loader(train_path: str, test_path: str = None,
                   batch_size: int = 10, first_n_photos: int = 0, manual_seed: int = 42):

    train_transform = transforms.Compose([
        transforms.RandomRotation(10),
        transforms.RandomHorizontalFlip(),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    test_transform = transforms.Compose([
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    train_data = datasets.ImageFolder(train_path, transform=train_transform)
    test_data = datasets.ImageFolder(test_path, transform=test_transform)

    torch.manual_seed(manual_seed)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

    if first_n_photos != 0:
        sample_loader = []
        train_loader_iter = iter(train_loader)

        for n in range(first_n_photos):
            sample_loader.append(next(train_loader_iter))
        # print(type(DataLoader(sample_loader)))
        return DataLoader(sample_loader), sample_loader

    return train_data, test_data, train_loader, test_loader


def get_number_of_classes(train_data_path):

    data = datasets.ImageFolder(train_data_path)

    return len(data.classes)


def read_to_loader_n_photos(data_path, n):
    test_transform = transforms.Compose([
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    data = datasets.ImageFolder(data_path, transform=test_transform)
    data = torch.utils.data.Subset(data, np.random.choice(len(data), n))

    loader = DataLoader(data)

    return data, loader


if __name__ == '__main__':
    read_to_loader_n_photos('/home/peprycy/WasteClassifier/Data/TrashNet', 5)
