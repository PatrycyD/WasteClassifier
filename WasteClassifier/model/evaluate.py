import pandas as pd

from network import ConvolutionalNetwork, HOGNeuralNetwork, get_inception_nn, load_inception_pickle
import torch
import WasteClassifier.config as config
from typing import List
from WasteClassifier.preprocessing.images_preprocessing import DataManager
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import numpy as np
import cv2
from torch.autograd import Variable
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns
import copy
import sys
np.set_printoptions(threshold=sys.maxsize)


class Evaluator(DataManager):  # class Evaluator(Transformer):

    def __init__(self, data_path, cnn, use_binary_models, transform_type: str = 'test',
                 batch_size: int = 10, grayscale: bool = True):
        super().__init__(data_path, cnn, transform_type, batch_size, grayscale)
        self.use_binary_models = use_binary_models
        self.model = None
        self.model_plastic = None
        self.model_glass = None
        self.model_cardboard = None
        self.model_metal = None
        self.model_organic = None
        self.all_models = None
        self.binary_order = None
        self.num_classes = None

    def load_model(self, model, pickle_path: str, num_classes: int):
        if hog_transform:
            pickle_path = pickle_path.replace('.pickle', '_hog.pickle')

        if self.use_binary_models:
            binary_order = config.classes

            self.model_cardboard = copy.deepcopy(model) #ConvolutionalNetwork(num_classes, config.channels) if hog_transformed \
                # else HOGNeuralNetwork(num_classes, config.hog_nn_input_params)
            cardboard_path = 'model_cardboard'.join(pickle_path.rsplit('model', 1))
            # print('model_cardboard'.join(pickle_path.rsplit('model', 1)))
            # self.model_cardboard.load_state_dict(torch.load('model_cardboard'.join(pickle_path.rsplit('model', 1))))
            if callable(getattr(self.model_cardboard, 'load_pickle', None)):
                self.model_cardboard.load_pickle(cardboard_path) #replace last occurence
            else:
                self.model_cardboard = load_inception_pickle(self.model_cardboard, cardboard_path)
            # self.model_cardboard.eval()

            self.model_glass = copy.deepcopy(model) #ConvolutionalNetwork(num_classes, config.channels) if hog_transformed \
                # else HOGNeuralNetwork(num_classes, config.hog_nn_input_params)
            glass_path = 'model_glass'.join(pickle_path.rsplit('model', 1))
            if callable(getattr(self.model_glass, 'load_pickle', None)):
                self.model_glass.load_pickle(glass_path)
            else:
                self.model_glass = load_inception_pickle(self.model_glass, glass_path)
            # self.model_glass.eval()

            # self.model_metal = ConvolutionalNetwork(1, config.channels)
            # self.model_metal.load_pickle(config.model_metal_pickle_path)
            # self.model_metal.eval()

            self.model_organic = copy.deepcopy(model)  # ConvolutionalNetwork(num_classes, config.channels) if hog_transformed \
                # else HOGNeuralNetwork(num_classes, config.hog_nn_input_params)
            organic_path = 'model_organic'.join(pickle_path.rsplit('model', 1))
            if callable(getattr(self.model_organic, 'load_pickle', None)):
                self.model_organic.load_pickle(organic_path)
            else:
                self.model_organic = load_inception_pickle(self.model_organic, organic_path)
            # self.model_organic.eval()

            self.model_plastic = copy.deepcopy(model)  # ConvolutionalNetwork(num_classes, config.channels) if hog_transformed \
                # else HOGNeuralNetwork(num_classes, config.hog_nn_input_params)
            plastic_path = 'model_plastic'.join(pickle_path.rsplit('model', 1))
            if callable(getattr(self.model_plastic, 'load_pickle', None)):
                self.model_plastic.load_pickle(plastic_path)
            else:
                self.model_plastic = load_inception_pickle(self.model_plastic, plastic_path)
            # self.model_plastic.eval()

            self.all_models = [self.model_cardboard, self.model_glass, #self.model_metal,
                               self.model_organic, self.model_plastic]

        else:
            self.model = torch.load('/home/peprycy/WasteClassifier/WasteClassifier/model/model_full_model.pickle')
            # self.model.fc = torch.nn.Sequential(
            #                                     torch.nn.Linear(self.model.fc.in_features, 10),
            #                                     torch.nn.ReLU(),
            #                                     torch.nn.Dropout(0.3),
            #                                     torch.nn.Linear(10, num_classes),
            #                                     torch.nn.Softmax(dim=1)
            #                                     )
            # print(self.model)
            # self.model = copy.deepcopy(model) # ConvolutionalNetwork(num_classes, config.channels) if hog_transformed \
            #     else HOGNeuralNetwork(num_classes, config.hog_nn_input_params)
            # if callable(getattr(self.model, 'load_pickle', None)):
            #     self.model.load_pickle(pickle_path)

            # self.model.load_state_dict(torch.load(pickle_path))
            # self.model.eval()

    def forward(self, input_image):

        if self.batch_size == 1:
            input_image = torch.unsqueeze(input_image.type(torch.FloatTensor), 0)

        input_image_wrapped = Variable(input_image)

        feed_forward_results = self.model.forward(input_image_wrapped.squeeze())
        (prob, class_label) = torch.max(feed_forward_results, 1)
        return class_label.data, prob, feed_forward_results

    def forward_binary(self, input_image):
        for net in self.all_models:
            net.eval()
        if self.batch_size == 1:
            input_image = torch.unsqueeze(input_image.type(torch.FloatTensor), 0)

        input_image_wrapped = Variable(input_image)

        if self.batch_size == 1:
            probs = [net.forward(input_image_wrapped.squeeze()).item() for net in self.all_models]
            predicted_class = probs.index(max(probs))
            probability = max(probs)

            return predicted_class, probability

        else:
            if self.num_classes == 1:
                probs = [net.forward(input_image_wrapped) for net in self.all_models]
            else:
                probs = [torch.max(net.forward(input_image_wrapped), dim=1)[0] for net in self.all_models]

            predicted_classes = torch.tensor([])
            for idx in range(self.batch_size):
                one_object_preds = torch.tensor([prob[idx].item() for prob in probs])
                predicted_classes = torch.cat((predicted_classes, torch.argmax(one_object_preds).reshape(-1, 1)), 1)

            probabilities = torch.tensor([])
            for idx in range(self.batch_size):
                one_object_preds = torch.tensor([prob[idx].item() for prob in probs])
                probabilities = torch.cat((probabilities, torch.max(one_object_preds).reshape(1, -1)))
            # predicted_classes[predicted_classes == 1.] = 8.
            # predicted_classes[predicted_classes == 0.] = 1.
            # predicted_classes[predicted_classes == 8.] = 0.
            return predicted_classes, probabilities

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

        x_tr = torch.from_numpy(x_tr_features)
        y_tr = torch.from_numpy(y.reshape(-1, 1))
        y_tr = y_tr.type(torch.LongTensor)
        return x_tr, y_tr

    def evaluate_dataset(self, metrics: List[str], display_photos: bool = True):
        if self.dataloader is None and self.image_folder is None:
            self.return_dataset_and_loader(shuffle=True)

        with torch.no_grad():
            correct = 0
            missed = 0
            all_predictions = np.array([])
            y_trues = np.array([])
            raw_predictions = np.array([]).reshape(-1, self.num_classes)
            for X, y in self.dataloader:
                if X.shape[0] != self.batch_size:
                    break

                if hog_transform:
                    orig_X, orig_y = X, y
                    X, y_train = self.convert_input_data(X, y)
                    y = y_train.float().squeeze(1)

                X = X.float()

                prediction, prob, raw_nn_output = self.forward(X) if not self.use_binary_models else self.forward_binary(X)
                raw_predictions = np.vstack((raw_predictions, raw_nn_output))

                correct += (prediction == y).sum()
                missed += (prediction != y).sum()
                if config.batch_size > 1:
                    prediction = prediction.squeeze()
                    for idx in range(config.batch_size):
                        all_predictions = np.append(all_predictions, prediction[idx])
                        y_trues = np.append(y_trues, y[idx])
                else:
                    all_predictions = np.append(all_predictions, prediction)
                    y_trues = np.append(y_trues, y)

                if display_photos:
                    for idx in range(self.batch_size):
                        predicted_class_name = int(all_predictions[idx])
                        predicted_class_name = config.classes[predicted_class_name]
                        true_class_name = int(orig_y[idx].item())
                        true_class_name = config.classes[true_class_name]
                        im = make_grid(orig_X[idx])
                        plt.figure(figsize=(15, 12))
                        plt.title(f'Predicted: {predicted_class_name}, true: {true_class_name}')
                        plt.imshow(np.transpose(im.numpy(), (1, 2, 0)))
                        plt.show()

        for metric in metrics:
            dataset_type = self.data_path.split('/')[-1]
            if metric == 'accuracy':
                accuracy_value = round(accuracy_score(y_trues.astype(int), all_predictions.astype(int)) * 100, 2)
                print(f'{dataset_type} accuracy value is {accuracy_value}%')

            elif metric == 'confusion_matrix':
                conf_mat = confusion_matrix(y_trues.astype(int), all_predictions.astype(int))
                print(f'{dataset_type} confiustion matrix')
                print(conf_mat)

        results_root_path = f'{config.model_root_path}/analytics'
        with open(f'{results_root_path}/{dataset_type}_raw_predictions', 'w') as file:
            file.write(str(raw_predictions))

        with open(f'{results_root_path}/{dataset_type}_true_values', 'w') as file:
            file.write(str(y_trues))

        with open(f'{results_root_path}/{dataset_type}_predictions', 'w') as file:
            file.write(str(all_predictions))

    @staticmethod
    def _map_labels(y_labels, batch_pred):
        pred_labels = [y_labels[pred.value] for pred in batch_pred]

        return torch.tensor([config.classes_dict[label] for label in pred_labels])


def main(saved_model_path, use_binary_models):

    train_path = f'{config.PREPROCESSED_IMAGES_PATH}/train'
    test_path = f'{config.PREPROCESSED_IMAGES_PATH}/test'

    test_evaluator = Evaluator(test_path, config.architecture,
                               use_binary_models, 'test', config.batch_size, config.grayscale)

    if use_binary_models and config.architecture == 'inception':
        num_classes = 2
    elif use_binary_models:
        num_classes = 1
    else:
        num_classes = test_evaluator.get_number_of_classes()

    test_evaluator.num_classes = num_classes

    if hog_transform:
        test_evaluator.load_model(HOGNeuralNetwork(num_classes, config.hog_nn_input_params),
                                  saved_model_path, num_classes)
    elif config.architecture == 'inception':
        test_evaluator.load_model(get_inception_nn(num_classes, saved_model_path), saved_model_path, num_classes)
    else:
        test_evaluator.load_model(ConvolutionalNetwork(num_classes, config.channels), saved_model_path, num_classes)

    test_evaluator.evaluate_dataset(metrics=['accuracy', 'confusion_matrix'], display_photos=False)

    train_evaluator = Evaluator(train_path, config.architecture,
                                use_binary_models, 'test', config.batch_size, config.grayscale)

    if use_binary_models and config.architecture == 'inception':
        num_classes = 2
    elif use_binary_models:
        num_classes = 1
    else:
        num_classes = train_evaluator.get_number_of_classes()

    train_evaluator.num_classes = num_classes

    if hog_transform:
        train_evaluator.load_model(HOGNeuralNetwork(num_classes, config.hog_nn_input_params),
                                   saved_model_path, num_classes)
    elif config.architecture == 'inception':
        train_evaluator.load_model(get_inception_nn(num_classes, False, saved_model_path), saved_model_path, num_classes)
        # model = torch.load('/home/peprycy/WasteClassifier/WasteClassifier/model/model_full_model.pickle')
    else:
        train_evaluator.load_model(ConvolutionalNetwork(num_classes, config.channels), saved_model_path, num_classes)

    train_evaluator.evaluate_dataset(metrics=['accuracy', 'confusion_matrix'], display_photos=False)


if __name__ == '__main__':
    use_binary_models = config.binary_train
    hog_transform = config.hog_transformation
    main(config.model_pickle_path, use_binary_models)
#TODO algorytym o centralizacji i skalowania zdjec - znalezc prace do rozpoznawania znakow drogowych,
# tam powinny byc informacje o tym jakie algorytmy sluza do przeskalowywana obrazow