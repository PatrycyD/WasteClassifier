import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import os
import cv2
import WasteClassifier.config as config
import shutil
import pathlib
import skimage
import random
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")


def prepare_dataset(source_path: str, target_path: str, depth: int = 2, hog_transformed: bool = True):
    target_path = pathlib.Path(target_path)
    source_path = pathlib.Path(source_path)

    prepare_directories(source_path, target_path)
    prepare_dirs_for_binary_models(target_path)

    if depth == 0:
        count = 0

        for file_name in source_path.iterdir():
            target_file_path = target_path / file_name.name
            img = file_name
            img = resize_file(img)
            img = resize_file(file_name)
            img = bgr_to_hsv(img)
            # img = hog_image(img)
            count += 1

    # I know it hurts, but this it's just python file manipulation stuff
    elif depth == 2:  # depth 2 is directory given in structure .../train|test/label/contents
        count = 0

        # first iterate over datasets split into train and test images
        for dataset in source_path.iterdir():
            # iterate over labels in train / test
            for label_path in dataset.iterdir():
                # iterate over every image in label directories
                for file_name in pathlib.Path(label_path).iterdir():
                    target_file_path = target_path / dataset.name / label_path.name / file_name.name
                    img = cv2.imread(str(file_name))

                    if img is None:
                        print(f'File {file_name} is broken. Removing')
                        pathlib.Path(file_name).unlink()
                        continue

                    img = convert_img_to_nn_input(img, hog_transformed, hsv_transformed=config.is_hsv)

                    if hog_transformed:
                        skimage.io.imsave(str(target_file_path), img)  # img.astype(np.uint8))
                        # skimage.io.imsave(str(target_file_path), img.astype(np.uint8))
                    else:
                        cv2.imwrite(str(target_file_path), img)

                    save_imgs_to_binary_catalogs(img, target_file_path.parents[2], label_path.name, file_name.name,
                                                 hog_transformed)
                    count += 1

                print(f'Extracting photos for {dataset.name} {label_path.name} done')

        balance_binary_datasets(target_path)


def convert_img_to_nn_input(img, hog_transformed, hsv_transformed):
    # if img.shape != (512, 384, 3):
    #     img = resize_file(img)

    if hsv_transformed:
        img = bgr_to_hsv(img)

    if hog_transformed:
        img = transform_with_hog(img)

    return img


def bgr_to_hsv(img):
    # input is cv2 image
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return img


def transform_with_hog(img):
    # skimage_img = cv2_to_skimage(img)
    # features, hog = skimage.feature.hog(img, orientations=9, pixels_per_cell=(8, 8),
    #                               cells_per_block=(2, 2), visualize=True, multichannel=True)
    fd, hog = skimage.feature.hog(img, orientations=9, pixels_per_cell=(8, 8),
                                  cells_per_block=(2, 2), visualize=True, multichannel=True)

    return hog


def cv2_to_skimage(img):
    img = img[:, :, ::-1]  # in opencv photos are BGR
    return skimage.util.img_as_float(img)


def skimage_to_cv2(img):
    img = img[:, :, ::-1]
    return skimage.util.img_as_ubyte(img)


def resize_file(img):
    # input is cv2 image
    resized = cv2.resize(img, (config.PHOTO_WIDTH, config.PHOTO_HEIGHT))
    return resized


def prepare_directories(src_dir, tgt_dir):

    # cannot use one function to delete all files under directory like rm -rf
    if tgt_dir.is_dir() and len([x for x in tgt_dir.iterdir()]) != 0:
        shutil.rmtree(tgt_dir)
    elif tgt_dir.is_dir():
        pathlib.Path(tgt_dir).rmdir()

    pathlib.Path(f'{tgt_dir}/train').mkdir(parents=True, exist_ok=False)
    pathlib.Path(f'{tgt_dir}/test').mkdir(parents=True, exist_ok=False)
    for label in os.listdir(f'{src_dir}/train'):
        pathlib.Path(f'{tgt_dir}/train/{label}').mkdir()
        pathlib.Path(f'{tgt_dir}/test/{label}').mkdir()


def prepare_dirs_for_binary_models(target_path):
    # binary classification assumption is to make separate models for every waste fraction, hence we need to have two
    # catalogs for every label this is: "label" and "everything but label"
    train_labels = [label_path for label_path in pathlib.Path(target_path, 'train').iterdir()]
    test_labels = [label_path for label_path in pathlib.Path(target_path, 'test').iterdir()]
    labels = train_labels + test_labels
    for label in labels:
        catalog_path = pathlib.Path(label.parents[1], f'{label.parent.name}_all_but_{label.name}')
        catalog_path.mkdir()
        pathlib.Path(catalog_path, label.name).mkdir()
        pathlib.Path(catalog_path, f'not_{label.name}').mkdir()


def save_imgs_to_binary_catalogs(img, target_path, label, filename, use_skimage):
    # function distributes images to all_but_dirs - for x label: in all_but_{x} puts to catalog {x} and in all_but_{y}
    # puts to not_{y} catalog
    not_dirs = [path for path in target_path.iterdir()
                if label not in path.name and path.name != 'train' and path.name != 'test']

    proper_dirs = [path for path in target_path.iterdir()
                   if label in path.name and path.name != 'train' and path.name != 'test']

    for all_but_dir in not_dirs:
        not_dir = f'{all_but_dir}/not_{all_but_dir.name.split("_")[-1]}'
        not_dir_file_path = f'{not_dir}/{filename}'
        if use_skimage:
            skimage.io.imsave(str(not_dir_file_path), img)#img.astype(np.uint8))
            # skimage.io.imsave(str(not_dir_file_path), img.astype(np.uint8))
        else:
            cv2.imwrite(not_dir_file_path, img)

    for proper_dir in proper_dirs:
        label_dir = f'{proper_dir}/{proper_dir.name.split("_")[-1]}'
        file_path = f'{label_dir}/{filename}'
        if use_skimage:
            skimage.io.imsave(str(file_path), img)  # img.astype(np.uint8))
            # skimage.io.imsave(str(file_path), img.astype(np.uint8))
        else:
            cv2.imwrite(file_path, img)


def balance_binary_datasets(target_path):
    # in order to binary dataset not being unbalanced gonna trim {not_label} dir to the length of {label} dir

    for binary_dataset in target_path.iterdir():
        if binary_dataset.name == 'train' or binary_dataset.name == 'test':
            continue

        label_path = binary_dataset / binary_dataset.name.split('_')[-1]
        all_but_path = binary_dataset / f'not_{binary_dataset.name.split("_")[-1]}'
        files_in_label_catalog = len(list(label_path.iterdir()))
        files_in_but_catalog = len(list(all_but_path.iterdir()))

        if files_in_label_catalog >= files_in_but_catalog:
            continue

        indexes_to_keep = random.sample(range(files_in_but_catalog), files_in_label_catalog)

        idx = -1
        for img in all_but_path.iterdir():
            idx += 1
            if idx in indexes_to_keep:
                continue
            else:
                img.unlink()


class DataManager:

    def __init__(self, data_path, cnn: 'string', transform_type: str = 'test',
                 batch_size: int = 10, grayscale: bool = True):
        self.data_path = data_path
        self.batch_size = batch_size
        self.grayscale = grayscale
        self.num_of_classes = None
        self.dataloader = None
        self.image_folder = None
        if self.grayscale:
            # norm_mean = 0.485
            norm_mean = 0.3276
            # norm_std = 0.229
            norm_std = 0.1824
        else:
            # norm_mean = [0.485, 0.456, 0.406]
            norm_mean = [0.2200, 0.0655, 0.0440]
            # norm_std= [0.229, 0.224, 0.225]
            norm_std = [0.1078, 0.0844, 0.0549]

        nn_input_size = 299 if cnn == 'incpetion' else (384, 512)

        if transform_type == 'test':
            transforms_list = [
                               transforms.Resize(nn_input_size),
                               transforms.CenterCrop(nn_input_size),
                               transforms.ToTensor(),
                               transforms.Normalize(norm_mean, norm_std)
                               ]
        else:
            transforms_list = [
                              transforms.Resize(nn_input_size),
                              transforms.RandomRotation(45),
                              transforms.RandomHorizontalFlip(),
                              transforms.RandomVerticalFlip(),
                              transforms.CenterCrop(nn_input_size),  # 384, 512
                              transforms.ToTensor(),
                              transforms.Normalize(norm_mean, norm_std)
                              ]
        if self.grayscale:
            transforms_list.append(transforms.Grayscale())

        self.transform = transforms.Compose(transforms_list)

    def return_dataset_and_loader(self, manual_seed: int = 42, return_loader: bool = True, shuffle: bool = True):

        data = datasets.ImageFolder(self.data_path, transform=self.transform)

        torch.manual_seed(manual_seed)
        if return_loader:
            loader = DataLoader(data, batch_size=self.batch_size, shuffle=shuffle)
            self.dataloader = loader
            self.image_folder = data
            return loader, data

        self.image_folder = data
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
    # read_to_loader_n_photos('/home/peprycy/WasteClassifier/Data/TrashNet', 5)
    prepare_dataset(config.SPLIT_IMAGES_PATH, config.PREPROCESSED_IMAGES_PATH, 2, hog_transformed=False)
