import copy
import torch
from network import ConvolutionalNetwork, HOGNeuralNetwork, get_inception_nn
import WasteClassifier.config as config
from WasteClassifier.preprocessing.images_preprocessing import DataManager
import os
import cv2
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score


class Trainer:
    def __init__(self, dataset_path, batch_size, binary_train=None):

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
        train_manager = DataManager(self.train_path, architecture, transform_type='train',
                                    batch_size=self.batch_size, grayscale=config.grayscale)
        train_loader, train_data = train_manager.return_dataset_and_loader(shuffle=True)

        # mean = torch.zeros(config.channels)
        # std = torch.zeros(config.channels)
        # print('==> Computing mean and std..')
        # length = 0
        # for b,(inputs, _labels) in enumerate(train_loader):
        #     for i in range(config.channels):
        #         length += 1
        #         # print(mean.shape)
        #         mean[i] += inputs[:, i, :, :].mean()
        #         std[i] += inputs[:, i, :, :].std()
        # mean.div_(length)
        # std.div_(length)
        # print('mean and standard deviation are')
        # print(mean, std)

        if self.binary_train:
            self.num_of_classes = 1 if architecture != 'inception' else 2
        else:
            self.num_of_classes = train_manager.get_number_of_classes()

        test_manager = DataManager(self.test_path, architecture, transform_type='test',
                                   batch_size=self.batch_size, grayscale=config.grayscale)
        test_loader, test_data = test_manager.return_dataset_and_loader()

        self.train_data = train_data
        self.test_data = test_data
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.classes = train_data.classes

        return train_data, test_data, train_loader, test_loader

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

    # @staticmethod
    def convert_input_data(self, X, y):
        X = X.detach().numpy()
        batched = True if X.shape[0] > 1 or len(X.shape) < 4 else False
        y = y.detach().numpy()
        cell_size = config.cell_size_params
        block_size = config.block_size_params
        bins = config.bins_params
        if batched:
            x_tr_features = np.array([])
            for record in X:
                single_x_feature = self.compute_hog(cell_size, block_size, bins, record)
                x_tr_features = x_tr_features.reshape(-1, single_x_feature.shape[1])
                x_tr_features = np.vstack((x_tr_features, single_x_feature))
        else:
            x_tr_features = self.compute_hog(cell_size, block_size, bins, X.squeeze(0))
        # x_ev_features = self.compute_hog(cell_size_params[i], block_size_params[q], bins_params[r], x_eval)

        x_tr = torch.from_numpy(x_tr_features)
        y_tr = torch.from_numpy(y.reshape(-1, 1))
        y_tr = y_tr.type(torch.LongTensor)

        return x_tr, y_tr

    def get_svm_input(self):
        for b, (X_train, y_train) in enumerate(self.train_loader):

            if hog_transform:
                X_train, y_train = self.convert_input_data(X_train, y_train)
                y_train = y_train.float().squeeze(1)

    def train(self, model, criterion, optimizer, epochs=config.epochs, count_time=False, verbose=False):

        if count_time:
            import time
            start_time = time.time()

        epochs = epochs
        max_trn_batch = 800
        train_losses = []
        train_correct = []
        test_accuracies = []
        for i in range(epochs):
            trn_corr = 0

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
                b += 1

                if hog_transform:
                    X_train, y_train = self.convert_input_data(X_train, y_train)
                    y_train = y_train.float().squeeze(1)

                X_train = X_train.float()
                y_pred = model.forward(X_train)

                if self.num_of_classes == 1:
                    y_pred = y_pred.reshape(y_pred.shape[0])
                    y_train = y_train.float()

                if self.num_of_classes > 1:
                    predicted = torch.max(y_pred, 1)[1]
                else:
                    predicted = torch.round(y_pred)
                batch_corr = (predicted == y_train).sum()
                trn_corr += batch_corr

                # Update parameters
                optimizer.zero_grad()
                loss = criterion(y_pred, y_train)
                loss.backward()
                optimizer.step()

                # Print interim results
                if b % 75 == 0:
                    print(f'epoch: {i+1:2}  batch: {b:4} [{self.batch_size * b:6}/{self.total_photos_num}]  \
                    loss: {loss.item():10.8f}  \
                    train accuracy: {trn_corr.item() * 100 / (self.batch_size * b):7.3f}%')

            train_losses.append(loss)
            train_correct.append(trn_corr)

            # Run the testing batches
            with torch.no_grad():
                all_predictions = np.array([])
                y_trues = np.array([])
                for X, y in self.test_loader:
                    if X.shape[0] != self.batch_size:
                        break

                    if hog_transform:
                        X, y_train = self.convert_input_data(X, y)
                        y = y_train.float().squeeze(1)

                    X = X.float()

                    val_pred = model.forward(X)
                    if self.num_of_classes > 1:
                        val_predicted = torch.max(val_pred, 1)[1]
                    else:
                        val_predicted = torch.round(val_pred)

                    val_predicted = val_predicted.squeeze()
                    for idx in range(config.batch_size):
                        all_predictions = np.append(all_predictions, val_predicted[idx])
                        y_trues = np.append(y_trues, y[idx])

            accuracy_value = round(accuracy_score(y_trues.astype(int), all_predictions.astype(int)) * 100, 2)
            test_accuracies.append(accuracy_value)
            conf_mat = confusion_matrix(y_trues.astype(int), all_predictions.astype(int))
            print(f'Epoch {i+1} test accuracy: {accuracy_value}%')
            print(conf_mat)

        if count_time:
            total_time = time.time() - start_time
            print(f'Training took {round(total_time / 60, 2)} minutes')

        return model, train_losses, test_accuracies, all_predictions, y_trues


def main(save_model_path=None, binary_train=None):

    dataset_path = f'{config.PREPROCESSED_IMAGES_PATH}'

    trainer = Trainer(dataset_path, config.batch_size, binary_train)
    trainer.get_data_loaders()
    print(trainer.binary_train)

    if hog_transform:
        model = HOGNeuralNetwork(trainer.num_of_classes, config.hog_nn_input_params)
    elif architecture == 'inception':
        model = get_inception_nn(trainer.num_of_classes, True, None)
    else:
        model = ConvolutionalNetwork(trainer.num_of_classes, config.channels)

    criterion = eval(config.loss_function) if trainer.num_of_classes > 1 else eval(config.binary_loss_function)
    optimizer = eval(config.optimizer)
    (model, train_losses, test_accuracies, \
    predictions, true_values) = trainer.train(model, criterion, optimizer, config.epochs, True)

    if save_model_path is not None:

        if trainer.binary_train:
            save_model_path = save_model_path.replace('model.pickle', f'model_{trainer.binary_train}.pickle')

        if hog_transform:
            save_model_path = save_model_path.replace('.pickle', '_hog.pickle')
        print(save_model_path)
        torch.save(model.state_dict(), save_model_path)
        torch.save(model, save_model_path.replace('.pickle', '_full_model.pickle')) # save full model

    results_root_path = f'{config.model_root_path}/analytics'
    with open(f'{results_root_path}/train_losses', 'w') as file:
        file.write(str(train_losses))
    with open(f'{results_root_path}/test_accuracies', 'w') as file:
        file.write(str(test_accuracies))


if __name__ == '__main__':
    hog_transform = config.hog_transformation
    binary_train = config.binary_train
    architecture = config.architecture
    if binary_train:
        for label in config.classes:
            main(config.model_pickle_path, label)
    else:
        main(config.model_pickle_path, None)
