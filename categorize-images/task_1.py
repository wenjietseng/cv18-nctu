""" Task 1: Tiny image representation
    - Simply resizes each image to a small, fixed resolution (16*16).
    - You can either resize the images to square while ignoring their aspect ratio or you can crop 
      the center square portion out of each image.
    - The entire image is just a vector of 16*16 = 256 dimensions.
    - Nearest neighbor classifier
"""

import numpy as np
import cv2
import os, sys


# Read Data
train_path = "./hw4_data/train/"
train_img_list = []
train_img_dirs = [d for d in os.listdir(train_path) if not d.startswith('.')]
img_names = [os.listdir(train_path + d) for d in train_img_dirs if not d.startswith('.')]

for d_idx in range(len(img_names)):
    for name in img_names[d_idx]:
        if not name.startswith('.'):
            # print(train_path + train_img_dirs[d_idx] + '/' + name)
            img = cv2.imread(train_path + train_img_dirs[d_idx] + '/' + name, 0)
            train_img_list.append(img)

print("Number of training images: %d " % len(train_img_list))

test_path = "./hw4_data/test/"
test_img_list = []
test_img_dirs = [d for d in os.listdir(test_path) if not d.startswith('.')]
img_names = [os.listdir(test_path + d) for d in test_img_dirs if not d.startswith('.')]

for d_idx in range(len(img_names)):
    for name in img_names[d_idx]:
        if not name.startswith('.'):
            # print(test_path + test_img_dirs[d_idx] + '/' + name)
            img = cv2.imread(test_path + test_img_dirs[d_idx] + '/' + name, 0)
            test_img_list.append(img)
print("Number of testing images: %d " % len(test_img_list))





# https://github.com/bikz05/bag-of-words