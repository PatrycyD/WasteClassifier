import torch
import network
import WasteClassifier.config as config
from WasteClassifier.preprocessing.images_preprocessing import DataManager
import os
import cv2
import skimage
import numpy as np
import matplotlib.pyplot as plt


class Trainer:
    def __init__(self, dataset_path, batch_size, binary_train=None):
        # self.model = model

        if binary_train:
            self.binary_train = binary_train
            self.train_path = f'{dataset_path}/train_all_but_{binary_train}'
            self.test_path = f'{dataset_path}/test_all_but_{binary_train}'
        else:
            self.binary_train = None
            self.train_path = f'{dataset_path}/train'
            self.test_path = f'{dataset_path}/test'

        self.batch_size = batch_size

        self.total_photos_num = 0
        for directory in os.listdir(self.train_path):
            self.total_photos_num += len(next(os.walk(f'{self.train_path}/{directory}'))[2])

        binaries_allowlist = ['plastic', 'metal', 'glass', 'cardboard', 'organic']
        if binary_train not in binaries_allowlist and binary_train is not None:
            raise ValueError('Model can be trained only on "plasitc", "metal", "glass", "cardboard" or "organic"')

        self.train_data = None
        self.test_data = None
        self.train_loader = None
        self.test_loader = None
        self.sample_loader = None
        self.sample_data = None
        self.num_of_classes = None
        self.classes = None

    def get_data_loaders(self, data_sample: int = 0):

        train_manager = DataManager(self.train_path, transform_type='train', batch_size=self.batch_size)
        train_loader, train_data = train_manager.return_dataset_and_loader()
        if self.binary_train:
            self.num_of_classes = 1
        else:
            self.num_of_classes = train_manager.get_number_of_classes()

        test_manager = DataManager(self.test_path, transform_type='test', batch_size=self.batch_size)
        test_loader, test_data = test_manager.return_dataset_and_loader()

        self.train_data = train_data
        self.test_data = test_data
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.classes = train_data.classes

        return train_data, test_data, train_loader, test_loader

    @staticmethod
    def convert_batch_to_hog(img):

        # skimage_img = cv2_to_skimage(img)
        # # features, hog = skimage.feature.hog(img, orientations=9, pixels_per_cell=(8, 8),
        # #                               cells_per_block=(2, 2), visualize=True, multichannel=True)
        # features = skimage.feature.hog(skimage_img, orientations=9, pixels_per_cell=(8, 8),
        #                                cells_per_block=(2, 2), visualize=False, multichannel=True)
        # # print(hog.shape)
        # print('skimage hog')
        # print(features.shape)
        if len(img.shape) == 4:  # if batched
            # img = img.reshape((img.shape[0], img.shape[2], img.shape[3], img.shape[1]))
            img = img.numpy()
        else:
            # img = img.reshape((img.shape[2], img.shape[3], img.shape[1]))
            img.numpy()

        # print(type(img[0]))
        # print(img[0].shape)
        image = cv2.imread('/home/peprycy/WasteClassifier/Data/custom_images_resized/20220327_133310.jpg')
        # print(type(image))
        # print(image.shape)
        winSize = (384, 512)
        blockSize = (16, 16)
        blockStride = (8, 8)
        cellSize = (8, 8)
        nbins = 9
        derivAperture = 1
        winSigma = 4.
        histogramNormType = 0
        L2HysThreshold = 2.0000000000000001e-01
        gammaCorrection = 0
        nlevels = 64
        # hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins, derivAperture, winSigma,
        #                         histogramNormType, L2HysThreshold, gammaCorrection, nlevels)
        hog = cv2.HOGDescriptor()
        # compute(img[, winStride[, padding[, locations]]]) -> descriptors
        winStride = (8, 8)
        padding = (8, 8)
        locations = ((10, 20),)
        if len(img.shape) == 4:  # if batched
            batch_hist = np.array([])
            for x in img:
                print(x.shape)
                print(type(x))

                # grayimg = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
                # print(grayimg.shape)
                # cv2.imshow('name', x)
                hist = hog.compute(x) #, winStride, padding, locations)
                batch_hist = np.append(batch_hist, hist)

            hist = batch_hist

        else:
            hist = hog.compute(img, winStride, padding, locations)
        print(type(hist))
        print(hist.shape)
        return hist

    def train(self, model, criterion, optimizer, epochs=config.epochs, count_time=False, verbose=False):

        if count_time:
            import time
            start_time = time.time()

        epochs = epochs
        max_trn_batch = 800
        max_tst_batch = 300
        train_losses = []
        test_losses = []
        train_correct = []
        test_correct = []

        for i in range(epochs):
            trn_corr = 0
            tst_corr = 0

            for b, (X_train, y_train) in enumerate(self.train_loader):

                if b == max_trn_batch:
                    break
                self.convert_batch_to_hog(X_train)
                b += 1
                y_pred = model.forward(X_train)
                y_pred = y_pred.reshape(y_pred.shape[0])
                # print(y_pred.reshape(10))
                # print(y_pred.reshape(10).shape)
                # print(y_train.float())
                # print(y_train.shape)
                y_train = y_train.float()
                loss = criterion(y_pred, y_train)
                # print(y_pred)
                # print(torch.round(y_pred))
                if not self.binary_train:
                    predicted = torch.max(y_pred, 1)[1]
                else:
                    predicted = torch.round(y_pred)
                # print('predicted')
                # print(predicted)
                # print('y_train')
                # print(y_train)
                batch_corr = (predicted == y_train).sum()
                # print(batch_corr / len(predicted))
                trn_corr += batch_corr

                # Update parameters
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Print interim results
                if b % 75 == 0:
                    print(f'epoch: {i+1:2}  batch: {b:4} [{self.batch_size * b:6}/{self.total_photos_num}]  \
                    loss: {loss.item():10.8f}  \
                    accuracy: {trn_corr.item() * 100 / (self.batch_size * b):7.3f}%')

            train_losses.append(loss)
            train_correct.append(trn_corr)

            # Run the testing batches
            with torch.no_grad():
                for b, (X_test, y_test) in enumerate(self.test_loader):
                    # Limit the number of batches
                    if b == max_tst_batch:
                        break
                    y_test = y_test.float()
                    # Apply the model
                    y_val = model.forward(X_test)
                    y_val = y_val.reshape(y_val.shape[0])

                    # Tally the number of correct predictions
                    if not self.binary_train:
                        predicted = torch.max(y_val.data, 1)[1]
                    else:
                        predicted = torch.round(y_val)

                    tst_corr += (predicted == y_test).sum()

                    loss = criterion(y_val, y_test)
                    test_losses.append(loss)
                    test_correct.append(tst_corr)

        if count_time:
            total_time = time.time() - start_time
            print(f'Training took {round(total_time / 60, 2)} minutes')

        return model

    @staticmethod
    def hog_transformation(tensor_img):
        hog = cv2.HOGDescriptor()
        im = cv2.imread(sample)
        h = hog.compute(im)
        return h


def main(save_model_path=None, binary_train=None):

    dataset_path = f'{config.PREPROCESSED_IMAGES_PATH}'

    trainer = Trainer(dataset_path, config.batch_size, binary_train)
    trainer.get_data_loaders()

    model = network.ConvolutionalNetwork(trainer.num_of_classes)
    # model.add_classes(trainer.train_data)

    criterion = eval(config.loss_function) if not binary_train else eval(config.binary_loss_function)
    optimizer = eval(config.optimizer)
    model = trainer.train(model, criterion, optimizer, config.epochs, True)

    if trainer.binary_train:
        save_model_path = save_model_path.replace('model.pickle', f'model_{trainer.binary_train}.pickle')

    if save_model_path is not None:
        torch.save(model.state_dict(), save_model_path)


if __name__ == '__main__':
    for label in config.classes:
        main(config.model_pickle_path, label)
    # main(config.model_pickle_path, 'plastic')
