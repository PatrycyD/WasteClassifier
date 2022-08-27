import torch
import network
import WasteClassifier.config as config
from WasteClassifier.preprocessing.images_preprocessing import DataManager
import os
import cv2
import numpy as np


class Trainer:
    def __init__(self, dataset_path, batch_size, binary_train=None):
        # self.model = model

        if binary_train:
            self.binary_train = binary_train
            self.train_path = f'{dataset_path}/train_all_but_{binary_train}'
            self.test_path = f'{dataset_path}/test_all_but_{binary_train}'
            # print(self.train_path)
            # print(self.test_path)
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

        train_manager = DataManager(self.train_path, transform_type='train',
                                    batch_size=self.batch_size, grayscale=config.grayscale)
        train_loader, train_data = train_manager.return_dataset_and_loader(shuffle=True)
        if self.binary_train:
            self.num_of_classes = 1
        else:
            self.num_of_classes = train_manager.get_number_of_classes()

        test_manager = DataManager(self.test_path, transform_type='test',
                                   batch_size=self.batch_size, grayscale=config.grayscale)
        test_loader, test_data = test_manager.return_dataset_and_loader()

        self.train_data = train_data
        self.test_data = test_data
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.classes = train_data.classes

        return train_data, test_data, train_loader, test_loader

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

            stop_train_switch_file = f'{config.project_root_path}/WasteClassifier/model/stop_training_flag'
            with open(stop_train_switch_file, 'r') as file:
                stop_train_switch = file.read().strip('\n')

            if stop_train_switch == '1':
                print('Stop training flag trigerred')
                with open(stop_train_switch_file, 'w') as file:
                    file.write('0')
                break

            for b, (X_train, y_train) in enumerate(self.train_loader):

                if b == max_trn_batch:
                    break
                if hog_transform:
                    x_tr_features = self.compute_hog(cell_size_params[i], block_size_params[q], bins_params[r], x_train)
                    x_ev_features = compute_hog(cell_size_params[i], block_size_params[q], bins_params[r], x_eval)

                    x_tr = torch.from_numpy(x_tr_features).to(device)
                    y_tr = torch.from_numpy(y_train.reshape(-1, 1))
                    y_tr = y_tr.type(torch.LongTensor).to(device)

                    x_ev = torch.from_numpy(x_ev_features).to(device)
                    y_ev = torch.from_numpy(y_eval.reshape(-1, 1))
                    y_ev = y_ev.type(torch.LongTensor).to(device)
                b += 1
                y_pred = model.forward(X_train)
                # print(y_pred.shape)
                if self.binary_train is not None:
                    y_pred = y_pred.reshape(y_pred.shape[0])
                    y_train = y_train.float()
                # print(y_pred)
                # print(y_pred.reshape(10).shape)
                # print(y_train)
                # print(y_train.shape)
                loss = criterion(y_pred, y_train)
                # print(y_pred)
                # print(torch.round(y_pred))
                if not self.binary_train:
                    predicted = torch.max(y_pred, 1)[1]
                    # print('predicted')
                    # print(predicted)
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

                    # Apply the model
                    y_val = model.forward(X_test)
                    if self.binary_train is not None:
                        y_val = y_val.reshape(y_val.shape[0])
                        y_test = y_test.float()

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
    def compute_hog(cell_size, block_size, nbins, imgs_gray):
        """
        Function computes HOG features for images data using parameters
        Args:
            cell_size (tuple):  number of pixels in a square cell in x and y direction (e.g. (4,4), (8,8))
            block_size (tuple) : number of cells in a block in x and y direction (e.g., (1,1), (1,2))
            nbins (tuple) : number of bins in a orientation histogram in x and y direction (e.g. 6, 9, 12)
            imgs_gray (np.ndarray) : images with which to perform HOG feature extraction (dimensions (nr, width, height))
        Returns:
            hog_feats (np.ndarray) : array of shape H x imgs_gray.shape[0] where H is the size of the resulting HOG feature vector
        """
        hog = cv2.HOGDescriptor(_winSize=(imgs_gray.shape[2] // cell_size[1] * cell_size[1],
                                          imgs_gray.shape[1] // cell_size[0] * cell_size[0]),
                                _blockSize=(block_size[1] * cell_size[1],
                                            block_size[0] * cell_size[0]),
                                _blockStride=(cell_size[1], cell_size[0]),
                                _cellSize=(cell_size[1], cell_size[0]),
                                _nbins=nbins)
        # winSize is the size of the image cropped to a multiple of the cell size

        hog_example = hog.compute(np.squeeze(imgs_gray[0, :, :]).astype(np.uint8)).flatten().astype(np.float32)

        hog_feats = np.zeros([imgs_gray.shape[0], hog_example.shape[0]])

        for img_idx in range(imgs_gray.shape[0]):
            hog_image = hog.compute(np.squeeze(imgs_gray[img_idx, :, :]).astype(np.uint8)).flatten().astype(np.float32)
            hog_feats[img_idx, :] = hog_image

        return hog_feats


def main(save_model_path=None, binary_train=None):

    dataset_path = f'{config.PREPROCESSED_IMAGES_PATH}'

    trainer = Trainer(dataset_path, config.batch_size, binary_train)
    trainer.get_data_loaders()

    model = network.ConvolutionalNetwork(trainer.num_of_classes, config.channels)

    criterion = eval(config.loss_function) if not binary_train else eval(config.binary_loss_function)
    optimizer = eval(config.optimizer)
    model = trainer.train(model, criterion, optimizer, config.epochs, True)

    if trainer.binary_train:
        save_model_path = save_model_path.replace('model.pickle', f'model_{trainer.binary_train}.pickle')

    if save_model_path is not None:
        torch.save(model.state_dict(), save_model_path)


if __name__ == '__main__':
    hog_transform = config.hog_transformation
    binary_train = config.binary_train
    if binary_train:
        for label in config.classes:
            main(config.model_pickle_path, label)
    else:
        main(config.model_pickle_path, None)
