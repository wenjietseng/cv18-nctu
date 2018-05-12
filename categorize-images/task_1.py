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
from sklearn.neighbors import KNeighborsClassifier

# Read Data
train_path = "./hw4_data/train/"
train_img_list = []
img_dirs = [d for d in os.listdir(train_path) if not d.startswith('.')]

img_names = [os.listdir(train_path + d) for d in img_dirs if not d.startswith('.')]

for d_idx in range(len(img_names)):
    for name in img_names[d_idx]:
        if not name.startswith('.'):
            # print(train_path + img_dirs[d_idx] + '/' + name)
            img = cv2.imread(train_path + img_dirs[d_idx] + '/' + name, 0)
            train_img_list.append(img)

print("Number of training images: %d " % len(train_img_list))

test_path = "./hw4_data/test/"
test_img_list = []
img_names = [os.listdir(test_path + d) for d in img_dirs if not d.startswith('.')]

for d_idx in range(len(img_names)):
    for name in img_names[d_idx]:
        if not name.startswith('.'):
            # print(test_path + img_dirs[d_idx] + '/' + name)
            img = cv2.imread(test_path + img_dirs[d_idx] + '/' + name, 0)
            test_img_list.append(img)
print("Number of testing images: %d " % len(test_img_list))

# --- Preparing Data ---
# resize to 16 x 16 and flatten
resized_imgs = [cv2.resize(img, (16, 16), interpolation=cv2.INTER_CUBIC) for img in train_img_list]
Xtr = np.asarray([img.flatten() for img in resized_imgs], dtype=float)
resized_imgs = [cv2.resize(img, (16, 16), interpolation=cv2.INTER_CUBIC) for img in test_img_list]
Xte = np.asarray([img.flatten() for img in resized_imgs], dtype=float)

# labels
label_dict = {}
for i, c in enumerate(img_dirs):
    label_dict[i] = c

Ytr = np.repeat(np.arange(15), 100)
Yte = np.repeat(np.arange(15), 10)

# Nearest Neighbor
class NearestNeighbor(object):
    def __init__(self):
        pass
    
    def train(self, X, y):
        """ X is N x 256 where each row is an example. Y is 1-dimension of size N """
        # the nearest neighbor classifier simply remembers all the training data
        self.Xtr = X
        self.ytr = y

    def predict(self, X, dist_type='L2'):
        """ X is N x 256 where each row is an example we wish to predict label for """
        num_test = X.shape[0]
        # lets make sure that the output type matches the input type
        Ypred = np.zeros(num_test, dtype = self.ytr.dtype)
        if dist_type == 'L2':
            for i in range(num_test):
                distances = np.sqrt(np.sum(np.square(self.Xtr - X[i,:]), axis = 1))
                min_index = np.argmin(distances)
                Ypred[i] = self.ytr[min_index]

        elif dist_type == 'L1':
            # loop over all test rows
            for i in range(num_test):
                # find the nearest training image to the i'th test image
                # using the L1 distance (sum of absolute value differences)
                distances = np.sum(np.abs(self.Xtr - X[i,:]), axis = 1)
                min_index = np.argmin(distances) # get the index with smallest distance
                Ypred[i] = self.ytr[min_index] # predict the label of the nearest example
        else:
            raise("False distance type")
        return Ypred

nn = NearestNeighbor()
nn.train(Xtr, Ytr)
Yte_predict = nn.predict(Xte, 'L1')
print('L1 accuracy: %f' % (np.mean(Yte_predict == Yte)))
Yte_predict = nn.predict(Xte, 'L2')
print('L2 accuracy: %f' % (np.mean(Yte_predict == Yte)))
# scikit: knn
 
 
# cross validation: use index on Xtr, Xte, Ytr, Yte



# knn.fit()
# visualization, confusion matrix

# https://github.com/bikz05/bag-of-words