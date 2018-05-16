from __future__ import print_function
import os
import sys
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

import torchvision
import torchvision.transforms as transforms
import torchvision.utils as vutils

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import PIL

from models import WJNet


# 1. Loading images and preprocessing (center crop, resize, normalizing, padding zero, random flip)

# 2. Define network, models are stored in model.py

# 3. Define loss function and optimizer

# Repeat 100 epochs
#   4. Training

#   5. Testing with test data

# 6. Output results

