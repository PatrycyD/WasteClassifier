from network import ConvolutionalNetwork
import torch
import WasteClassifier.config as config
from typing import List
from WasteClassifier.preprocessing.images_preprocessing import DataManager
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os
from torch.autograd import Variable


class Evaluator(DataManager):  # class Evaluator(Transformer):
    """
    Possible metrics:
    - accuracy,
    - kappa
    """
    available_metrics = {
        'accuracy': 'correct / (correct + missed)',
        'kappa': '(correct - len(correct + missed)/len(set(correct))) / (1 - len(corret + missed)/len(set(correct)))'
        ''' 
        kappa - kappa statistic: (acc - acc_random) / (1 - acc_random)
         acc is accuracy of the model, acc_random are random classes assignments
        '''
    }

    def __init__(self, data_path, use_binary_models, transform_type: str = 'test',
                 batch_size: int = 10, grayscale: bool = True):
        super().__init__(data_path, transform_type, batch_size, grayscale)
        self.use_binary_models = use_binary_models
        self.model = None
        self.model_plastic = None
        self.model_glass = None
        self.model_cardboard = None
        self.model_metal = None
        self.model_organic = None
        self.all_models = None
        self.binary_order = None

    def load_model(self, model, pickle_path, num_classes):
        if self.use_binary_models:
            binary_order = config.classes

            self.model_cardboard = ConvolutionalNetwork(1, config.channels)
            self.model_cardboard.load_pickle(config.model_cardboard_pickle_path)
            self.model_cardboard.eval()

            self.model_glass = ConvolutionalNetwork(1, config.channels)
            self.model_glass.load_pickle(config.model_glass_pickle_path)
            self.model_glass.eval()

            # self.model_metal = ConvolutionalNetwork(1, config.channels)
            # self.model_metal.load_pickle(config.model_metal_pickle_path)
            # self.model_metal.eval()

            self.model_organic = ConvolutionalNetwork(1, config.channels)
            self.model_organic.load_pickle(config.model_organic_pickle_path)
            self.model_organic.eval()

            self.model_plastic = ConvolutionalNetwork(1, config.channels)
            self.model_plastic.load_pickle(config.model_plastic_pickle_path)
            self.model_plastic.eval()

            self.all_models = [self.model_cardboard, self.model_glass, #self.model_metal,
                               self.model_organic, self.model_plastic]

        else:
            self.model = ConvolutionalNetwork(num_classes, config.channels)
            self.model.load_state_dict(torch.load(config.model_pickle_path))
            self.model.eval()

    def load_binary_models(self):
        pass

    def forward(self, input_image):

        if self.batch_size == 1:
            input_image = torch.unsqueeze(input_image.type(torch.FloatTensor), 0)

        input_image_wrapped = Variable(input_image)

        feed_forward_results = self.model.forward(input_image_wrapped)
        # print(feed_forward_results)
        (prob, class_label) = torch.max(feed_forward_results, 1)
        # print(class_label)
        return prob, class_label.data[0]

    def forward_binary(self, input_image):
        for net in self.all_models:
            net.eval()
        if self.batch_size == 1:
            input_image = torch.unsqueeze(input_image.type(torch.FloatTensor), 0)

        input_image_wrapped = Variable(input_image)
        # print(input_image_wrapped.shape)
        # print(input_image_wrapped.item())

        if self.batch_size == 1:
            probs = [net(input_image_wrapped).item() for net in self.all_models]
            predicted_class = probs.index(max(probs))
            probability = max(probs)

            return predicted_class, probability. probs

        else:
            probs = [net(input_image_wrapped) for net in self.all_models]

            # print(probs)
            # print(type(probs))
            predicted_classes = torch.tensor([])
            for idx in range(self.batch_size):
                one_object_preds = torch.tensor([prob[idx].item() for prob in probs])
                # print(torch.argmax(one_object_preds).reshape(-1, 1)
                # print(one_object_preds)
                predicted_classes = torch.cat((predicted_classes, torch.argmax(one_object_preds).reshape(-1, 1)), 1)
                # print(predicted_classes)
            # predicted_classes = [torch.argmax(prob) for prob in probs]

            probabilities = torch.tensor([])
            for idx in range(self.batch_size):
                one_object_preds = torch.tensor([prob[idx].item() for prob in probs])
                # print(torch.max(one_object_preds).reshape(1, -1))
                probabilities = torch.cat((probabilities, torch.max(one_object_preds).reshape(1, -1)))
            # print(predicted_classes)
            predicted_classes[predicted_classes == 1.] = 8.
            predicted_classes[predicted_classes == 0.] = 1.
            predicted_classes[predicted_classes == 8.] = 0.
            return predicted_classes, probabilities

    def evaluate_dataset(self, metrics: List[str], display_photos: bool = True):
        if self.dataloader is None and self.image_folder is None:
            self.return_dataset_and_loader(shuffle=True)

        if self.image_folder.classes != list(config.classes_dict.keys()):
            classes = list(config.classes_dict.keys())
            classes_dict = {}
        else:
            classes = self.image_folder.classes

        with torch.no_grad():
            correct = 0
            missed = 0
            all_predictions = np.array([])
            for X, y in self.dataloader:
                # print(y)
                # print(type(X))
                # print(X.shape)
                if X.shape[0] != self.batch_size:
                    break
                prediction, prob = self.forward(X) if not self.use_binary_models else self.forward_binary(X)
                # print(prediction)
                # print(prob)
                # prediction, prob = self.forward_binary()
                # print(prediction)
                # print(predictions)
                # predicted_class = torch.max(predictions, 1)[1] if self.use_bunary_models else XXXX
                # print('preds')
                # print(predicted_class)
                # print('true')
                # print(y)
                # y_true = _map_labels(image_folder.classes, y)
                correct += (prediction == y).sum()
                missed += (prediction != y).sum()
                for pred in prediction:
                    all_predictions = np.append(all_predictions, pred)

                if display_photos:
                    for idx in range(self.batch_size):
                        predicted_class_name = int(all_predictions[idx])
                        predicted_class_name = config.classes[predicted_class_name]
                        true_class_name = y[idx]
                        true_class_name = config.classes[true_class_name]
                        im = make_grid(X[idx])
                        plt.figure(figsize=(15, 12))
                        plt.title(f'Predicted: {predicted_class_name}, true: {true_class_name}')
                        plt.imshow(np.transpose(im.numpy(), (1, 2, 0)))
                        plt.show()

        for metric in metrics:
            metric_value = eval(Evaluator.available_metrics[metric])
            metric_value = round(metric_value.item(), 2)
            print(f'{metric} value is {metric_value}')
        # print(all_predictions)

    @staticmethod
    def _map_labels(y_labels, batch_pred):
        pred_labels = [y_labels[pred.value] for pred in batch_pred]

        return torch.tensor([config.classes_dict[label] for label in pred_labels])


def main(saved_model_path, use_binary_models):

    train_path = f'{config.PREPROCESSED_IMAGES_PATH}/train'
    test_path = f'{config.PREPROCESSED_IMAGES_PATH}/test'

    test_evaluator = Evaluator(test_path, use_binary_models, 'test', config.batch_size, config.grayscale)
    num_classes = test_evaluator.get_number_of_classes()
    test_evaluator.load_model(ConvolutionalNetwork(num_classes, config.channels), saved_model_path, num_classes)
    test_evaluator.evaluate_dataset(metrics=['accuracy'], display_photos=True)

    train_evaluator = Evaluator(train_path, use_binary_models, 'test', config.batch_size, config.grayscale)
    num_classes = train_evaluator.get_number_of_classes()
    train_evaluator.load_model(ConvolutionalNetwork(num_classes, config.channels), saved_model_path, num_classes)
    train_evaluator.evaluate_dataset(metrics=['accuracy'], display_photos=False)


if __name__ == '__main__':
    use_binary_models = config.binary_train
    main(config.model_pickle_path, use_binary_models)
#TODO algorytym o centralizacji i skalowania zdjec - znalezc prace do rozpoznawania znakow drogowych,
# tam powinny byc informacje o tym jakie algorytmy sluza do przeskalowywana obrazow