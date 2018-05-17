from __future__ import print_function
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.autograd import Variable

import torchvision
import torchvision.transforms as transforms
import torchvision.utils as vutils

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from models import WJNet
from utils import *

# 1. Loading images and preprocessing (center crop, resize, normalizing, padding zero, random flip)
print('====> Loading Data ')
train_dataset = ImageDataset(root_dir='../hw4_data', mode='train',
                             transform=transforms.Compose([
                                        transforms.CenterCrop(220),
                                        transforms.Resize(254),
                                        transforms.Pad(1, fill=0),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5,), (0.5,)),
                             ]))
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)

test_dataset = ImageDataset(root_dir='../hw4_data', mode='test',
                            transform=transforms.Compose([
                                      transforms.CenterCrop(220),
                                      transforms.Resize(256), 
                                      transforms.ToTensor(), 
                                      transforms.Normalize((0.5,), (0.5,)),
                            ]))
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)

classes = [d for d in os.listdir('../hw4_data/train') if not d.startswith('.')]

print('====> Complete loading data!')

# 2. Define network, models are stored in model.py
net = WJNet()

# 3. Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 60], gamma=0.1)

# 4. Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()

    train_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (inputs, labels) in enumerate(train_loader):

        print(len(inputs))
        print(len(labels))
        print(labels)
        # zero the parameter gradients
        optimizer.zero_grad()

        inputs, labels = Variable(inputs), Variable(labels)
        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        train_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += predicted.eq(labels.data).cpu().sum()

        if batch_idx % 9 == 0:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.5f' %
                  (epoch + 1, batch_idx + 1, train_loss / 10))
            running_loss = 0.0


# 5. Testing with test data
def test(epoch):
    pass

# Repeat 100 epochs
for epoch in range(80):
    train(epoch)
    test(epoch)

# 6. Output results