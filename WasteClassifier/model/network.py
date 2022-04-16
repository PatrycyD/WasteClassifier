import torch
import torch.nn.functional as F


class ConvolutionalNetwork(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.classes = None
        self.num_classes = num_classes
        self.conv1 = torch.nn.Conv2d(3, 16, 3, 1)
        self.conv2 = torch.nn.Conv2d(16, 32, 3, 1)
        self.conv3 = torch.nn.Conv2d(32, 54, 3, 1)
        self.fc1 = torch.nn.Linear(54 * 26 * 26, 250)
        self.fc2 = torch.nn.Linear(250, 120)
        self.fc3 = torch.nn.Linear(120, 84)
        self.fc4 = torch.nn.Linear(84, 60)
        self.fc5 = torch.nn.Linear(60, self.num_classes)

    # def add_classes(self, data):
    #     print(data.classes)
    #     self.classes = torch.nn.Parameter(torch.tensor(data.classes))

    def forward(self, x):
        x = F.relu(self.conv1(x))
        # print(x.shape)
        x = F.max_pool2d(x, 2, 2)
        # print(x.shape)
        x = F.relu(self.conv2(x))
        # print(x.shape)
        x = F.max_pool2d(x, 2, 2)
        # print(x.shape)
        x = F.relu(self.conv3(x))
        # print(x.shape)
        x = F.max_pool2d(x, 2, 2)
        # print(x.shape)
        x = x.reshape(-1, 54 * 26 * 26)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        # print(x)
        # x = F.log_softmax(x, dim=0)
        # print('softmax')
        # print(x)
        return torch.sigmoid(x)
