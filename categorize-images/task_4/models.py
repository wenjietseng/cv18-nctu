import torch.nn as nn
import torch.nn.functional as F

class WJNet(nn.Module):
    def __init__(self):
        super(WJNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, stride=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(16, 32, 5)
        self.pool3 = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(6 * 21 * 25, 128)
        self.fc2 = nn.Linear(128, 100)
        self.fc3 = nn.Linear(100, 15)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        print('conv1 finished')
        print(x.size())

        x = self.pool1(x)
        print('pool1 finish')
        print(x.size())
        x = F.relu(self.conv2(x))
        print('conv2 finish')
        print(x.size())
        x = self.pool2(x)
        print('pool2 finish')
        print(x.size())

        x = F.relu(self.conv3(x))
        print('conv3 finish')
        print(x.size())
        x = self.pool3(x)
        print('pool3 finish')
        print(x.size())
        # flatten
        x = x.view(-1, self.num_flat_features(x))
        print(x.size())
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

