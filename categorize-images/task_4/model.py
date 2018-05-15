import torch.nn as nn
import torch.nn.functional as F

class WJNet(nn.Module):
    def __init__(self):
        super(WJNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 4, 4)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(4, 16, 4)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16 * 4 * 4, 128)
        self.fc2 = nn.Linear(128, 100)
        self.fc3 = nn.Linear(100, 15)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        # flatten
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
net = WJNet()
print(net)