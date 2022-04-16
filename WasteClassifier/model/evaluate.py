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
from torchsummary import summary


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

    def __init__(self, data_path, transform_type: str = 'test', batch_size: int = 10):
        super().__init__(data_path, transform_type, batch_size)
        self.model = None

    def load_model(self, model, pickle_path):
        # self.model = model
        model.load_state_dict(torch.load(pickle_path))
        model.eval()
        self.model = model

    # TODO - do wywalenia przekazywanie dataloaderow - powinno je sciagac z atrybutow
    def evaluate_dataset(self, dataloader, image_folder, metrics: List[str], display_photos: bool = False):
        print(image_folder.classes)
        if image_folder.classes != list(config.classes_dict.keys()):
            classes = list(config.classes_dict.keys())
            classes_dict = {}
        else:
            classes = image_folder.classes

        with torch.no_grad():
            correct = 0
            missed = 0
            all_predictions = np.array([])
            for X, y in dataloader:
                # print(X)
                # print(type(X))
                # print(X[0].shape)
                predictions = self.model.forward(X)
                # print(predictions)
                predicted_class = torch.max(predictions, 1)[1]
                # print('preds')
                # print(predicted_class)
                # print('true')
                # print(y)
                # y_true = _map_labels(image_folder.classes, y)
                correct += (predicted_class == y).sum()
                missed += (predicted_class != y).sum()
                for pred in predicted_class:
                    all_predictions = np.append(all_predictions, pred)

        for metric in metrics:
            metric_value = eval(Evaluator.available_metrics[metric])
            metric_value = round(metric_value.item(), 2)
            print(f'{metric} value is {metric_value}')
        # print(all_predictions)

        if display_photos:
            for idx, (photo, label) in enumerate(dataloader):

                predicted = predictions[idx]  # tensor([2, 1, 0, 4, 3, 3])
                im = make_grid(photo)

                plt.figure(figsize=(15, 12))
                plt.title(f'Predicted: {all_predictions}, true: {label}')
                plt.imshow(np.transpose(im.numpy(), (1, 2, 0)))
                plt.show()

    def predict_classes(self, display_photos: bool = True):
        # print(summary(self.model, (3, 224, 224)))
        batch_count = 0
        for photo in os.listdir(self.data_path):
            batch_count += 1
            img = Image.open(f'{self.data_path}/{photo}')
            img_tensor = self.transform(img).reshape(1, 3, 224, 224)
            # print(img_tensor)
            # print(type(img_tensor))
            # print(img_tensor.shape)

            with torch.no_grad():
                prediction = self.model.forward(img_tensor)
                # print(prediction)
                prediction = torch.max(prediction, 1)[1]
                prediction = config.classes_dict[prediction.item()]

            if display_photos:
                im = make_grid(img_tensor)

                plt.figure(figsize=(15, 12))
                plt.title(f'Predicted: {prediction}')
                plt.imshow(np.transpose(im.numpy(), (1, 2, 0)))
                # plt.imshow(im)
                plt.show()

    @staticmethod
    def _map_labels(y_labels, batch_pred):
        pred_labels = [y_labels[pred.value] for pred in batch_pred]

        return torch.tensor([config.classes_dict[label] for label in pred_labels])


def main(saved_model_path):
    train_path = f'{config.split_images_path}/train'
    test_path = f'{config.split_images_path}/test'

    evaluator = Evaluator(test_path, 'test', config.batch_size)
    num_classes = evaluator.get_number_of_classes()
    evaluator.load_model(ConvolutionalNetwork(num_classes), saved_model_path)

    test_data_manager = DataManager(test_path, 'test', 10)
    test_loader, test_data = test_data_manager.return_dataset_and_loader()

    train_data_manager = DataManager(train_path, 'test', 10)
    train_loader, train_data = train_data_manager.return_dataset_and_loader()

    # evaluator.evaluate_dataset(train_loader, train_data, ['accuracy'])
    evaluator.evaluate_dataset(test_loader, test_data, ['accuracy'])

    custom_photos_predictor = Evaluator(config.resized_custom_photos_path, 'test', 1)
    num_classes = evaluator.get_number_of_classes()
    custom_photos_predictor.load_model(ConvolutionalNetwork(num_classes), saved_model_path)

    custom_photos_predictor.predict_classes(True)


if __name__ == '__main__':
    main(config.model_pickle_path)