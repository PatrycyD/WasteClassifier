import torch
import torch.nn.functional as F


class ConvolutionalNetwork(torch.nn.Module):
    def __init__(self, num_classes, channels):
        super().__init__()
        self.num_classes = num_classes
        self.channels = channels
        self.output_activation = torch.sigmoid if self.num_classes == 1 else torch.softmax
        self.classes = None
        self.conv1 = torch.nn.Conv2d(self.channels, self.channels * 6, 3, 1)
        self.conv2 = torch.nn.Conv2d(self.channels * 6, self.channels * 12, 3, 1)
        self.conv3 = torch.nn.Conv2d(self.channels * 12, self.channels * 20, 3, 1)
        self.fc1 = torch.nn.Linear((self.channels*20) * 46 * 62, 250)
        self.dropout = torch.nn.Dropout(0.25)
        self.fc2 = torch.nn.Linear(250, 120)
        self.fc3 = torch.nn.Linear(120, 84)
        self.fc4 = torch.nn.Linear(84, 60)
        self.fc5 = torch.nn.Linear(60, self.num_classes)

    def load_pickle(self, pickle_path):
        self.load_state_dict(torch.load(pickle_path))
        self.eval()

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.reshape(-1, (self.channels*20) * 46 * 62)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        # x = torch.sigmoid(x) if self.num_classes == 1 else torch.softmax(x, dim=1)
        x = torch.sigmoid(x)
        # x = self.output_activation(x)
        return x


class LeNet(torch.nn.Module):
    def __init__(self, num_classes, channels):
        super().__init__()
        self.num_classes = num_classes
        self.channels = channels
        self.output_activation = torch.sigmoid if self.num_classes == 1 else torch.softmax
        self.classes = None
        self.conv1 = torch.nn.Conv2d(self.channels, self.channels * 6, 3, 1)
        self.conv2 = torch.nn.Conv2d(self.channels * 6, self.channels * 12, 3, 1)
        self.conv3 = torch.nn.Conv2d(self.channels * 12, self.channels * 20, 3, 1)
        self.fc1 = torch.nn.Linear((self.channels * 20) * 46 * 62, 250)
        self.dropout = torch.nn.Dropout(0.25)
        self.fc2 = torch.nn.Linear(250, 120)
        self.fc3 = torch.nn.Linear(120, 84)
        self.fc4 = torch.nn.Linear(84, 60)
        self.fc5 = torch.nn.Linear(60, self.num_classes)


class HOGNeuralNetwork(torch.nn.Module):
    def __init__(self, num_classes, input_params):
        super().__init__()
        self.num_classes = num_classes
        self.output_activation = torch.sigmoid if self.num_classes == 1 else torch.softmax
        self.classes = None
        self.fc1 = torch.nn.Linear(input_params, 250)
        self.dropout = torch.nn.Dropout(0.25)
        self.fc2 = torch.nn.Linear(250, 120)
        self.fc3 = torch.nn.Linear(120, 84)
        self.fc4 = torch.nn.Linear(84, 60)
        self.fc5 = torch.nn.Linear(60, self.num_classes)

    def load_pickle(self, pickle_path):
        self.load_state_dict(torch.load(pickle_path))
        self.eval()

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        x = torch.sigmoid(x) if self.num_classes == 1 else torch.softmax(x, dim=1)
        # x = torch.sigmoid(x)
        # x = self.output_activation(x)
        return x


def get_inception_nn(num_classes: int, pretrained: bool, load_pickle_path: str = None):
    incpetion_v3 = torch.hub.load('pytorch/vision:v0.10.0', 'inception_v3', pretrained=pretrained)
    incpetion_v3.aux_logits = False

    for parameter in incpetion_v3.parameters():
        parameter.requires_grad = False

    incpetion_v3.fc = torch.nn.Sequential(
        torch.nn.Linear(incpetion_v3.fc.in_features, 1000),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.3),
        torch.nn.Linear(1000, 100),
        torch.nn.ReLU(),
        torch.nn.Linear(100, num_classes),
        torch.nn.Softmax(dim=1)
    )

    # if load_pickle_path is not None:
    #     incpetion_v3.load_state_dict(torch.load(load_pickle_path))
    #     incpetion_v3.eval()

    return incpetion_v3


def load_inception_pickle(model, pickle_path):
    model.load_state_dict(torch.load(pickle_path))
    model.eval()
    return model
