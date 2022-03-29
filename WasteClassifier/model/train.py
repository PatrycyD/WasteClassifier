import torch
import network
import config
from WasteClassifier.preprocessing.images_preprocessing import read_to_loader, get_number_of_classes
import pathlib


class Trainer:
    def __init__(self, train_path, test_path, batch_size):
        # self.model = model
        self.train_path = train_path
        self.test_path = test_path
        self.batch_size = batch_size
        self.train_data = None
        self.test_data = None
        self.train_loader = None
        self.test_loader = None
        self.sample_loader = None
        self.sample_data = None
        self.classes = None

    # def get_num_of_classes(self):
    #     assert self.train_data is not None, 'train loader should be defined earlier'
    #
    #     return len(self.train_data.classes)

    def get_data_loaders(self, data_sample: int = 0):

        if data_sample == 0 or data_sample is None:
            train_data, test_data, train_loader, test_loader = read_to_loader(self.train_path,
                                                                              self.test_path, self.batch_size)

            self.train_data = train_data
            self.test_data = test_data
            self.train_loader = train_loader
            self.test_loader = test_loader
            self.classes = train_data.classes

            return train_data, test_data, train_loader, test_loader

        else:
            sample_loader, sample_data = read_to_loader(self.train_path, batch_size=config.batch_size, first_n_photos=3)
            self.sample_loader = sample_loader
            self.sample_data = sample_data
            self.classes = sample_data.classes

            return sample_loader, sample_data

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
                    print(f'epoch: {i:2}  batch: {b:4} [{10 * b:6}/...]  loss: {loss.item():10.8f}  \
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

    # trashnet_train_path = pathlib.Path(__file__).parents[2].resolve().joinpath(
    #     'Data', 'TrashNet', 'split_images', 'train'
    # )
    trashnet_train_path = f'{config.trashnet_path}/train'
    trashnet_test_path = f'{config.trashnet_path}/test'
    # transhet_test_path = str(trashnet_train_path).replace('train', 'test')
    num_classes = get_number_of_classes(trashnet_train_path)

    model = network.ConvolutionalNetwork(num_classes)

    trainer = Trainer(trashnet_train_path, trashnet_test_path, config.batch_size)
    trainer.get_data_loaders()

    criterion = eval(config.loss_function)
    optimizer = eval(config.optimizer)
    model = trainer.train(model, criterion, optimizer, config.epochs, True)

    if save_model_path is not None:
        torch.save(model.state_dict(), save_model_path)

    # if perform_test:
    #     sample_loader, sample_data = trainer.get_data_loaders()


if __name__ == '__main__':
    main(save_model_path=config.model_pickle_path)
