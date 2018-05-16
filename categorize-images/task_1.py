""" Task 1: Tiny image representation
    - Simply resizes each image to a small, fixed resolution (16*16).
    - You can either resize the images to square while ignoring their aspect ratio or you can crop 
      the center square portion out of each image.
    - The entire image is just a vector of 16*16 = 256 dimensions.
    - Nearest neighbor classifier
    - KNN + Cross Validation
    - Confusion matrix visualization result
"""
import numpy as np
import cv2
import os, sys
from matplotlib import pyplot as plt

def crop_center(img):
    """ Crop the input image at center, the new size is the smaller one of original size
    Args:
        img - input image
    Return:
        a square size image which was cropped at center
    ref: https://stackoverflow.com/questions/39382412/crop-center-portion-of-a-numpy-image
    """
    h, w = img.shape
    if h > w:
        target_size = w
    elif h < w:
        target_size = h
    else:
        return img
    
    start_x = w//2 - (target_size//2)
    start_y = h//2 - (target_size//2)
    return img[start_y:start_y+target_size, start_x:start_x+target_size]

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

# --- Read Data ---
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
# crop at center
crop_train_imgs = map(crop_center, train_img_list)
crop_test_imgs = map(crop_center, test_img_list)

# resize to 16 x 16 and flatten
resized_imgs = [cv2.resize(img, (16, 16), interpolation=cv2.INTER_CUBIC) for img in train_img_list]
Xtr = np.asarray([img.flatten() for img in resized_imgs], dtype=float)

resized_imgs = [cv2.resize(img, (16, 16), interpolation=cv2.INTER_CUBIC) for img in test_img_list]
Xte = np.asarray([img.flatten() for img in resized_imgs], dtype=float)

# print(np.mean(Xtr, axis=0).shape)
# print(np.std(Xtr, axis=0).shape)
Xtr_normalized = (Xtr - np.mean(Xtr, axis=0)) / np.std(Xtr, axis=0)
Xte_normalized = (Xte - np.mean(Xte, axis=0)) / np.std(Xte, axis=0)

# print(np.mean(Xtr_normalized, axis=0))
# print(np.std(Xtr_normalized, axis=0))

# # labels
label_dict = {}
for i, c in enumerate(img_dirs):
    label_dict[i] = c

Ytr = np.repeat(np.arange(15), 100)
Yte = np.repeat(np.arange(15), 10)

# --- NN --- 
nn = NearestNeighbor()
nn.train(Xtr, Ytr)
Yte_predict_L1 = nn.predict(Xte, 'L1')
print('L1 accuracy: %f' % (np.mean(Yte_predict_L1 == Yte)))
Yte_predict_L2 = nn.predict(Xte, 'L2')
print('L2 accuracy: %f' % (np.mean(Yte_predict_L2 == Yte)))

nn.train(Xtr_normalized, Ytr)
Yte_predict_L1 = nn.predict(Xte_normalized, 'L1')
print('normalizd L1 accuracy: %f' % (np.mean(Yte_predict_L1 == Yte)))
Yte_predict_L2 = nn.predict(Xte_normalized, 'L2')
print('normalized L2 accuracy: %f' % (np.mean(Yte_predict_L2 == Yte)))

# confusion matrix
from sklearn.metrics import confusion_matrix
con_mat_L1 = confusion_matrix(Yte, Yte_predict_L1)
con_mat_L2 = confusion_matrix(Yte, Yte_predict_L2)

# Plot non-normalized confusion matrix
from plot_confusion_matrix import plot_confusion_matrix
plt.figure()
plot_confusion_matrix(con_mat_L1, classes=img_dirs,
                      title='Confusion matrix, NN')
plt.savefig('./task_1_out/confustion_mat_nn.png', bbox_inches='tight', dpi=300)
plt.close()

# --- KNN + cross validation ---
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# creating odd list of K for KNN
myList = list(range(1,50))

# subsetting just the odd ones
neighbors = filter(lambda x: x % 2 != 0, myList)

# empty list that will hold cv scores
cv_scores = []

# perform 10-fold cross validation
from sklearn.cross_validation import cross_val_score
for k in neighbors:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, Xtr_normalized, Ytr, cv=10, scoring='accuracy')
    cv_scores.append(scores.mean())

best_k = np.argmax(cv_scores) * 2 + 1
print('The best training acurracy of KNN (k=%d): %f' % (best_k, cv_scores[np.argmax(cv_scores)]))

# cv_scores acc plot with K
plt.figure()
plt.plot(np.arange(1, 50, step=2), cv_scores, color='black')
plt.axvline(x=best_k, linewidth=0.3, linestyle='--', color='firebrick')
plt.ylim(0, .4)
plt.xlabel('k from 1 to 49')
plt.ylabel('Accuracy')
plt.savefig('./task_1_out/knn-cv.png', bbox_inches='tight', dpi=300)

# test error with cross validation k
knn = KNeighborsClassifier(n_neighbors=best_k)
knn.fit(Xtr_normalized, Ytr)
knn_pred = knn.predict(Xte_normalized)
# print(knn_pred)
print('KNN acc (K = %d): %f' % (best_k, accuracy_score(Yte, knn_pred)))

# confusion mat
con_mat_knncv = confusion_matrix(Yte, knn_pred)
plt.figure()
plot_confusion_matrix(con_mat_knncv, classes=img_dirs,
                      title='Confusion matrix KNN (k=%d)' % best_k)
plt.savefig('./task_1_out/confustion_mat_knncv.png', bbox_inches='tight', dpi=300)
plt.close()
