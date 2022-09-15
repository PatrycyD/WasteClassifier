import numpy as np
import matplotlib.pyplot as plt
import WasteClassifier.config as config
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score
from torchmetrics import AUROC, ROC
import torch
import warnings
warnings.filterwarnings('ignore')


def read_files_values(results_root_path):
    with open(f'{results_root_path}/test_accuracies', 'r') as file:
        accuracies = file.read()
        accuracies = accuracies.replace('[', '').replace(']', '').replace(' ', '').split(',')
        accuracies = np.array([float(acc) for acc in accuracies])

    with open(f'{results_root_path}/train_losses', 'r') as file:
        losses = file.read()
        losses = losses.replace('[', '').replace(']', '').split(',')[::2]
        losses = np.array([float(loss.strip(' tensor(')) for loss in losses])

    with open(f'{results_root_path}/test_raw_predictions', 'r') as file:
        raw_predictions_evaluate_output = file.read()

    raw_preds = np.array([]).reshape(-1, len(raw_predictions_evaluate_output.split('\n')[0].split()))

    for record in raw_predictions_evaluate_output.split('\n'):
        raw_preds = np.vstack((raw_preds, record.replace('[', '').replace(']', '').split()))

    raw_preds = raw_preds.astype(float)

    with open(f'{results_root_path}/test_predictions', 'r') as file:
        preds_from_evaluate = file.read()
        preds_from_evaluate = preds_from_evaluate.replace('\n', '').replace('[', '').replace(']', '').split()
        preds_from_evaluate = np.array([float(pred) for pred in preds_from_evaluate])

    with open(f'{results_root_path}/test_true_values', 'r') as file:
        trues_from_evaluate = file.read()
        trues_from_evaluate = trues_from_evaluate.replace('[', '').replace(']', '').split()
        trues_from_evaluate = np.array([float(true) for true in trues_from_evaluate])

    return accuracies, losses, raw_preds, preds_from_evaluate, trues_from_evaluate


def generate_trues_and_preds_from_conf_mat(conf_mat):

    predictions = np.array([0. for x in range(np.sum(conf_mat, axis=0)[0])] +
                           [1. for x in range(np.sum(conf_mat, axis=0)[1])] +
                           [2. for x in range(np.sum(conf_mat, axis=0)[2])] +
                           [3. for x in range(np.sum(conf_mat, axis=0)[3])]
                           )

    true_values = np.array([])
    for i in conf_mat.transpose():
        trues_for_predicted_class = []
        for j in i:
            trues_for_predicted_class = trues_for_predicted_class + ([int(np.where(i == j)[0]) for x in range(j)])

        true_values = np.append(true_values, np.array(trues_for_predicted_class))

    true_values = true_values.reshape(true_values.shape[0], -1)
    predictions = predictions.reshape(predictions.shape[0], -1)

    # compare confision matrix downloaded from evaluate.py with confusion matrix generated from calculated above values
    # to validate if predictions and true values were correctly generated
    new_conf_mat = confusion_matrix(true_values, predictions)
    if not (new_conf_mat == conf_mat).all():
        raise ValueError('Confusion matrix copied from evaluate is different from newly generated confusion matrix.')

    return true_values, predictions, conf_mat


def generate_accuracy_over_epochs_chart(test_accuracies, epochs, save_path=None):
    fig = plt.figure(figsize=(20, 15))
    plt.plot(epochs, test_accuracies)
    plt.yticks(list(range(101)[::10]))
    plt.xticks(epochs[9::10])
    plt.title('accuracy value over epochs')
    plt.ylabel('accuracy value (%)')
    plt.xlabel('epoch')

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

    plt.close()


def generate_loss_over_epochs_chart(train_losses, epochs, save_path=None):
    fig = plt.figure(figsize=(20, 15))
    plt.plot(epochs, train_losses)
    # plt.yticks(list(range(101)[::10]))
    plt.xticks(epochs[9::10])
    plt.title('loss function value value over epochs')
    plt.ylabel('loss function value')
    plt.xlabel('epoch')

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

    plt.close()


def generate_confusion_matrix_heatmap(conf_mat, save_path=None):
    fig, ax = plt.subplots()
    ax.imshow(conf_mat)

    # Show all ticks and label them with the respective list entries
    ax.set_xticks(np.arange(len(config.classes)), labels=config.classes)
    ax.set_yticks(np.arange(len(config.classes)), labels=config.classes)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(len(config.classes)):
        for j in range(len(config.classes)):
            text = ax.text(j, i, conf_mat[i, j],  ha="center", va="center", color="w")

    ax.set_title("confusion matrix")
    fig.tight_layout()
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()


def one_hot_encode_array(lst, columns=None):
    columns = int(max(lst)) + 1 if not columns else columns
    arr = np.zeros((len(lst), columns), float)
    for i in range(len(lst)):
        arr[i, int(lst[i])] = 1.

    return arr


def calculate_roc_auc_score(true_values, predictions):
    true_values = torch.tensor(true_values).type(torch.int64)
    predictions = torch.from_numpy(predictions)

    auroc = AUROC(num_classes=4, average=None)
    all_auc_rocs = auroc(predictions, true_values)
    all_auc_rocs = [x.item() for x in all_auc_rocs]

    return all_auc_rocs


def plot_roc_curves(true_values, predictions, save_path=None):
    num_classes = int(max(true_values)+1)
    ns_probs = torch.tensor(one_hot_encode_array([0 for _ in range(len(true_values))], 4))
    true_values = torch.tensor(true_values).type(torch.int64)
    predictions = torch.from_numpy(predictions)
    roc = ROC(num_classes=4)
    fpr, tpr, thresholds = roc(predictions, true_values)
    ns_fpr, ns_tpr, _ = roc(ns_probs, true_values)
    # plt.figure(figsize=(20, 15))
    # after all it will only work for 4 classes, but leaving it anyways
    fig, ax = plt.subplots(int(num_classes/2), int(num_classes/2), figsize=(20, 15))
    idx = 0
    # colors = ['brown', 'green', 'blue', 'black']
    for i in range(int(num_classes/2)):
        for j in range(int(num_classes/2)):
            ax[i][j].set_title(config.classes[idx], fontsize=30)
            ax[i][j].plot(fpr[idx], tpr[idx], color='blue')
            ax[i][j].margins(0.01, 0.01)
            ax[i][j].tick_params(axis='both', which='major', labelsize=20)
            ax[i][j].plot(ns_fpr[idx], ns_tpr[idx], linestyle='--', color='royalblue')
            idx += 1

    plt.suptitle('ROC curves for each class', fontsize=30)

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()


def calculate_and_print_metrics(true_values, predictions, raw_predictions, conf_mat):
    each_class_precision = precision_score(true_values, predictions, average=None)
    each_class_recall = recall_score(true_values, predictions, average=None)
    each_class_roc = calculate_roc_auc_score(true_values, raw_predictions)

    for i in range(len(each_class_precision)):
        print(f'Precision score for {config.classes[i]}: {round(each_class_precision[i] * 100, 2)}%')
        print(f'Recall score for {config.classes[i]}: {round(each_class_recall[i] * 100, 2)}%')

        print(f'Roc Auc score for {config.classes[i]}: {round(each_class_roc[i], 2)}')

    print(f'Accuracy score for whole dataset: {round(accuracy_score(true_values, predictions)*100, 2)}%')
    print(f'Average precision score for all classes: {round((sum(each_class_precision) / len(each_class_precision)) * 100, 2)}%')
    print(f'Average recall score for all classes: {round((sum(each_class_recall) / len(each_class_recall)) * 100, 2)}%')
    print(f'Average roc auc score for all classes: {round(sum(each_class_roc) / len(each_class_roc), 2)}')
    print(f'Confusion matrix:\n{conf_mat}')


def main():

    results_root_path = f'{config.model_root_path}/analytics'
    (test_accuracies, train_losses, raw_predictions,
     predictions_from_evaluate, true_values_from_evaluate) = read_files_values(results_root_path)

    if not preds_and_trues_from_file:

        conf_mat = np.array([[538, 58, 31, 35], [25, 222, 19, 27],
                             [14, 13, 296, 4], [97, 102, 22, 117]])  # hardcoded output from evaluate script

        trues, preds, conf_mat = generate_trues_and_preds_from_conf_mat(conf_mat)

    else:
        trues = true_values_from_evaluate
        preds = predictions_from_evaluate
        conf_mat = confusion_matrix(trues, preds)

    calculate_and_print_metrics(trues, preds, raw_predictions, conf_mat)

    epochs = list(range(1, len(test_accuracies)+1))
    generate_accuracy_over_epochs_chart(test_accuracies, epochs, f'{results_root_path}/accuracy_chart.png')
    generate_loss_over_epochs_chart(train_losses, epochs, f'{results_root_path}/loss_chart.png')
    generate_confusion_matrix_heatmap(conf_mat, f'{results_root_path}/confusion_matrix.png')
    plot_roc_curves(trues, raw_predictions, f'{results_root_path}/ROC_curves')


if __name__ == '__main__':
    preds_and_trues_from_file = True
    main()
