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
        # print(self.fc5)

    def load_pickle(self, pickle_path):
        self.load_state_dict(torch.load(pickle_path))
        self.eval()

    def forward(self, x):
        # print(x.shape)
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
        # print(x)
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
