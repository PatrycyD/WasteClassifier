from network import ConvolutionalNetwork
import torch
import config
from typing import List
from sklearn.metrics import roc_auc_score
from WasteClassifier.preprocessing.images_preprocessing import read_to_loader, \
                        get_number_of_classes, read_to_loader_n_photos
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder


class Evaluator: # class Evaluator(Transformer):
    """
    Possible metrics:
    - accuracy,
    - kappa
    """
    available_metrics = {
        'accuracy': 'correct / missed',
        'kappa': '(correct - len(correct + missed)/len(set(correct))) / (1 - len(corret + missed)/len(set(correct)))'
        # kappa - kappa statistic. (acc - acc_random) / (1 - acc_random).
        # acc is accuracy of the model, acc_random are random classes assignments
    }
    test_transform = transforms.Compose([
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    train_data = datasets.ImageFolder(train_path)

    def __init__(self, model):
        self.model = model

    def evaluate_full_dataset(self, dataloader, metrics: List[str]):
        with torch.no_grad():
            correct = 0
            missed = 0
            for X, y in dataloader:
                predictions = self.model.forward(X)

                predicted_class = torch.max(predictions, 1)[1]
                # print('predictions')
                # print(predicted_class)
                # print('y_true')
                # print(y)

                correct += (predicted_class == y).sum()
                missed += (predicted_class != y).sum()

        for metric in metrics:
            metric_value = eval(Evaluator.available_metrics[metric])
            metric_value = round(metric_value.item(), 2)
            print(f'{metric} value is {metric_value}')

    def evaluate_first_n_photos(self, dataloader, n: int, metrics: List[str], display_photos: bool = True):
        import matplotlib.pyplot as plt

        dataloader = iter(dataloader)
        tmp_dataloader = []
        for image in range(n):
            tmp_dataloader.append(image)

        dataloader = DataLoader(tmp_dataloader)

        if display_photos:
            for image in dataloader:
                plt.imshow(image)
                plt.show()

        with torch.no_grad():
            correct = 0
            missed = 0
            for X, y in dataloader:
                predictions = self.model.forward(X)
                predicted_class = torch.max(predictions, 1)[1]
                correct += (predicted_class == y).sum()
                missed += (predicted_class != y).sum()

        for metric in metrics:
            metric_value = eval(Evaluator.available_metrics[metric])
            metric_value = round(metric_value.item(), 2)
            print(f'{metric} value is {metric_value}')

    def get_predictions(self, dataloader):
        pass

    def evaluate_custom_photos(self, custom_photos_path, display_photos: bool = True):
        dataloader = ImageFolder(custom_photos_path, transform=self.transform)



def main(saved_model_path):
    trashnet_train_path = f'{config.trashnet_path}/train'
    trashnet_test_path = f'{config.trashnet_path}/test'

    num_classes = get_number_of_classes(trashnet_train_path)
    model = ConvolutionalNetwork(num_classes)
    evaluator = Evaluator(model)
    train_data, test_data, train_loader, test_loader = read_to_loader(trashnet_train_path, trashnet_test_path)

    # evaluator.evaluate_full_dataset(train_loader, ['accuracy'])
    # evaluator.evaluate_full_dataset(test_loader, ['accuracy'])
    read_to_loader_n_photos('/home/peprycy/WasteClassifier/Data/TrashNet', 5)
    evaluator.evaluate_first_n_photos(train_loader, 5, ['accuracy'])


if __name__ == '__main__':
    main(config.model_pickle_path)