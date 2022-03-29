import torch
import torch.nn.functional as F


class ConvolutionalNetwork(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.test_data = None
        self.num_classes = num_classes
        self.conv1 = torch.nn.Conv2d(3, 6, 3, 1)
        self.conv2 = torch.nn.Conv2d(6, 16, 3, 1)
        self.fc1 = torch.nn.Linear(54 * 54 * 16, 120)
        self.fc2 = torch.nn.Linear(120, 84)
        self.fc3 = torch.nn.Linear(84, 60)
        self.fc4 = torch.nn.Linear(60, self.num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.reshape(-1, 54 * 54 * 16)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)

        return F.log_softmax(x, dim=0)
        # return F.relu(x)
