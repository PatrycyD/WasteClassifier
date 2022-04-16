import torch
import network
import WasteClassifier.config as config
from WasteClassifier.preprocessing.images_preprocessing import DataManager
import os


class Trainer:
    def __init__(self, train_path, test_path, batch_size):
        # self.model = model
        self.train_path = train_path
        self.test_path = test_path
        self.batch_size = batch_size

        self.total_photos_num = 0
        for directory in os.listdir(train_path):
            self.total_photos_num += len(next(os.walk(f'{train_path}/{directory}'))[2])

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
        self.num_of_classes = train_manager.get_number_of_classes()

        test_manager = DataManager(self.test_path, transform_type='test', batch_size=self.batch_size)
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
        print(epochs)
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
                # if b == max_trn_batch:
                #     break
                b += 1
                y_pred = model.forward(X_train)
                # print(y_pred)
                # print(y_train)
                loss = criterion(y_pred, y_train)

                predicted = torch.max(y_pred.data, 1)[1]
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

                    # Tally the number of correct predictions
                    predicted = torch.max(y_val.data, 1)[1]
                    tst_corr += (predicted == y_test).sum()

            loss = criterion(y_val, y_test)
            test_losses.append(loss)
            test_correct.append(tst_corr)

        if count_time:
            total_time = time.time() - start_time
            print(f'Training took {round(total_time / 60, 2)} minutes')

        return model


def main(save_model_path=None):

    train_path = f'{config.split_images_path}/train'
    test_path = f'{config.split_images_path}/test'

    trainer = Trainer(train_path, test_path, config.batch_size)
    trainer.get_data_loaders()

    model = network.ConvolutionalNetwork(trainer.num_of_classes)
    # model.add_classes(trainer.train_data)

    criterion = eval(config.loss_function)
    optimizer = eval(config.optimizer)
    model = trainer.train(model, criterion, optimizer, config.epochs, True)

    if save_model_path is not None:
        torch.save(model.state_dict(), save_model_path)


if __name__ == '__main__':
    main(save_model_path=config.model_pickle_path)
