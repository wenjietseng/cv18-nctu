from __future__ import print_function
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.autograd import Variable
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
import torchvision.utils as vutils

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from models import WJNet
from resnet import ResNet18
from utils import *
import torchvision.datasets as dset

import csv 
train_writer = csv.writer(open("./WJNet-train.csv", 'w'))
test_writer = csv.writer(open("./WJNet-test.csv", 'w'))
# 1. Loading images and preprocessing (center crop, resize, normalizing, padding zero, random flip)
my_transforms = transforms.Compose([transforms.Grayscale(),
                                    transforms.CenterCrop(220),
                                    transforms.Resize(222),
                                    transforms.Pad(1, fill=0),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,))])
print('====> Loading Data ')
train_dataset = dset.ImageFolder(root='../hw4_data/train',
                             transform=my_transforms)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
print(train_loader)
test_dataset = dset.ImageFolder(root='../hw4_data/test',
                            transform=my_transforms)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)

classes = [d for d in os.listdir('../hw4_data/train') if not d.startswith('.')]

print('====> Complete loading data!')

use_cuda = torch.cuda.is_available()

# 2. Define network, models are stored in model.py
net = WJNet() # 54% acc (16, 32, 64, 128)
# net = ResNet18()

if use_cuda:
    net.cuda()
    net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
    cudnn.benchmark = True

# 3. Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters())#, lr=0.1, momentum=0.9, weight_decay=1e-4)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 60], gamma=0.1)

# 4. Training
def train(epoch, writer):

    net.train()
    train_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (inputs, labels) in enumerate(train_loader):
        # print("in batch iters")
        # print(inputs.size())
        if use_cuda:
            inputs, labels = inputs.cuda(), labels.cuda()
        
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

        # if batch_idx % 10 == 0:    # print every 2000 mini-batches
        print('[%d, %5d] Loss: %.5f | Acc: %.3f (%d/%d)' %
                (epoch + 1, batch_idx + 1, train_loss / (batch_idx+1), 100.0*float(correct)/float(total), correct, total))
        writer.writerow([epoch, test_loss/(batch_idx+1),100.*correct/total])

# 5. Testing with test data
def test(epoch, writer):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(test_loader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        outputs = net(inputs)
        loss = criterion(outputs, targets)

        test_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        print('[%d, %5d] Loss: %.5f | Acc: %.3f%% (%d/%d)'
            % (epoch+1, batch_idx+1, test_loss/(batch_idx+1), 100.*float(correct)/float(total), correct, total))
        writer.writerow([epoch, test_loss/(batch_idx+1),100.*correct/total])

# Repeat 100 epochs
for epoch in range(80):
    print('\nEpoch: %d' % epoch)
    print('Training')
    train(epoch, train_writer)
    print('Testing')
    test(epoch, test_writer)

# 6. Output results