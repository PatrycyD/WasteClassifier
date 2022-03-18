import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import pathlib


def read_to_loader(batch_size: int=10):

    train_path = pathlib.Path(__file__).parents[2].resolve().joinpath(
                                                                        'Data',
                                                                        'TrashNet',
                                                                        'split_images',
                                                                        'train')

    test_path = str(train_path).replace('train', 'test')

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

    torch.manual_seed(42)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

    return train_data, test_data, train_loader, test_loader


if __name__ == '__main__':
    read_to_loader()
