import torch
import torch.nn as nn
import Model
import config
import WasteClassifier
from WasteClassifier.preprocessing.images_preprocessing import read_to_loader


def main():

    train_data, test_data, train_loader, test_loader = read_to_loader(config.batch_size)
    classes = test_data.classes
    model = Model.ConvolutionalNetwork(len(classes))
    model.test_data = test_data
    criterion = eval(config.loss_function)
    optimizer = eval(config.optimizer)
    print(criterion)

    torch.manual_seed(101)

    train(train_loader, test_loader, criterion, optimizer, model, config.epochs, count_time=True)


def train(train_loader, test_loader, criterion, optimizer, model,
          epochs=config.epochs, count_time=False, verbose=False):
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

        for b, (X_train, y_train) in enumerate(train_loader):
            # if b == max_trn_batch:
            #     break
            b += 1
            y_pred = model(X_train)
            loss = criterion(y_pred, y_train)

            predicted = torch.max(y_pred.data, 1)[1]
            batch_corr = (predicted == y_train).sum()
            trn_corr += batch_corr

            # Update parameters
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Print interim results
            if b % 35 == 0:
                print(f'epoch: {i:2}  batch: {b:4} [{10 * b:6}/8000]  loss: {loss.item():10.8f}  \
    accuracy: {trn_corr.item() * 100 / (10 * b):7.3f}%')

        train_losses.append(loss)
        train_correct.append(trn_corr)

        # Run the testing batches
        with torch.no_grad():
            for b, (X_test, y_test) in enumerate(test_loader):
                # Limit the number of batches
                if b == max_tst_batch:
                    break

                # Apply the model
                y_val = model(X_test)

                # Tally the number of correct predictions
                predicted = torch.max(y_val.data, 1)[1]
                tst_corr += (predicted == y_test).sum()

        loss = criterion(y_val, y_test)
        test_losses.append(loss)
        test_correct.append(tst_corr)

    if count_time:
        total_time = time.time() - start_time
        print(f'Training took {total_time / 60} minutes')


if __name__ == '__main__':
    main()
