import torch.nn as nn
import torch.nn.functional as F

class WJNet(nn.Module):
    def __init__(self):
        super(WJNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16 * 5 * 5, 128)
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
        # flatten
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
