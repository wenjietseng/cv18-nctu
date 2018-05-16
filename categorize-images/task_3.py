""" Task 3: Bag of SIFT representation with SVM
    - use cv2 to find SIFT feature descriptors, which should be Nx128 (N is the number of features detected)
    - Vector Quantization:
        Do K-means clustering to turn descriptors into groups
        historgram the grouping result, code them into another vector
        after this step, apply classifier as we did in task_1.py
    - SVM classifier
    - try with Cross Validation
    - Confusion matrix visualization result (see task_1.py)
"""

from sklearn import svm
from sklearn import cluster, metrics
import numpy as np
import cv2, os
import matplotlib.pyplot as plt

# Transform the clustered result into histogram
def clusterToHistogram(clt):
    # Create label from 0 to k
    numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)

    # Create histogram
    (hist, _) = np.histogram(clt.labels_, bins = numLabels)

    # Return histogram
    return hist

# Function used to read data
def dataRead():
    # Path to train data
    train_path = "./hw4_data/train/"
    train_img_list = []

    # Read the folder names
    img_dirs = [d for d in os.listdir(train_path) if not d.startswith('.')]

    # Read all the file names in folders, one folder by one folder0
    img_names = [os.listdir(train_path + d) for d in img_dirs if not d.startswith('.')]

    # Read all the images
    for d_idx in range(15):
        for name in img_names[d_idx]:
            if not name.startswith('.'):
                img = cv2.imread(train_path + img_dirs[d_idx] + '/' + name, 0)
                train_img_list.append(img)
   
    # Print out the number of training images
    # print("Number of training images: %d " % len(train_img_list))

    # Path to test data
    test_path = "./hw4_data/test/"
    test_img_list = []

    # Read all the file names in folders, one folder by one folder
    img_names = [os.listdir(test_path + d) for d in img_dirs if not d.startswith('.')]

    # Read all the images
    for d_idx in range(15):
        for name in img_names[d_idx]:
            if not name.startswith('.'):
                img = cv2.imread(test_path + img_dirs[d_idx] + '/' + name, 0)
                test_img_list.append(img)
    
    # Print out the number of training images
    # print("Number of training images: %d " % len(train_img_list))

    # Return data
    return img_dirs, train_img_list, test_img_list

# Generate bags of SIFT descriptors
def bagOfSIFT(train_img_list, test_img_list):
    # Create extractor
    extractor = cv2.xfeatures2d.SIFT_create()

    # TRAIN IMAGE
    # Compute descriptors, and store into descriptor array
    train_descriptors = []
    for img_idx in range(len(train_img_list)):
        _, descriptors = extractor.detectAndCompute(train_img_list[img_idx], None)
        train_descriptors.append(descriptors)

    # Do k-means clustering
    # Initiate KMeans
    clt = cluster.KMeans(n_clusters = 8, random_state = 0)

    # Cluster the histograms
    train_hist = []
    for clt_idx in range(len(train_descriptors)):
        clt.fit(train_descriptors[clt_idx])

        # Transform the clustered result into histogram
        train_hist.append(clusterToHistogram(clt))

    # TEST IMAGE
    # Compute descriptors, and store into descriptor array
    test_descriptors = []
    for img_idx in range(len(test_img_list)):
        _, descriptors = extractor.detectAndCompute(test_img_list[img_idx], None)
        test_descriptors.append(descriptors)

    # Cluster the histograms
    test_hist = []
    for clt_idx in range(len(test_descriptors)):
        clt.fit(test_descriptors[clt_idx])

        # Transform the clustered result into histogram
        test_hist.append(clusterToHistogram(clt))

    # Return histograms as np array
    return np.asarray(train_hist), np.asarray(test_hist)

# Read in all data
img_dirs, train_img_list, test_img_list = dataRead()

# Find descriptors and cluster. Returned in np array form
train_hist, test_hist = bagOfSIFT(train_img_list, test_img_list)

print(train_hist.shape)
print(test_hist.shape)


# Create lable dictionary
label_dict = {}
for idx, label in enumerate(img_dirs):
     label_dict[idx] = label

# Create Y results
train_Y = np.repeat(np.arange(15), 100)
test_Y = np.repeat(np.arange(15), 10)
    
# Show the histogram
# print(len(train_hist))
# print(len(test_hist))

#---------------------------------------------#
# SVM1
# trainData = train_hist
# Ytr = np.repeat(np.arange(15), 100)
# responses = Ytr.ravel()
# Yte = np.repeat(np.arange(15), 10)

# svm = cv2.ml.SVM_create()
# svm.setKernel(cv2.ml.SVM_LINEAR)
# svm.setType(cv2.ml.SVM_C_SVC)
# svm.setC(2.67)
# svm.setGamma(5.383)
# svm.train(trainData, cv2.ml.ROW_SAMPLE, responses)

# testData = test_hist
# result = svm.predict(testData)
# mask = result == responses
# correct = np.count_nonzero(mask)
# print(correct * 100.0 / result.size)


# SVM2
x_train = train_hist
y_train = train_Y
x_test = test_hist
y_test = test_Y

# clf = svm.SVC(C=0.8, kernel='rbf', gamma=20, decision_function_shape='ovr')
# clf.fit(x_train, y_train.ravel())
# print(clf.score(x_train, y_train))
# clf.score(x_train, y_train.ravel())
# print("clf.score:",clf.score)
# predicted = clf.predict(x_test)
# print("pre:",predicted)
X_train = train_hist
y_train = train_Y
X_test = test_hist
y_test = test_Y

svc_model = svm.SVC(gamma=0.001, C=100, kernel='linear')
svc_model.fit(X_train, y_train)
predicted = svc_model.predict(X_test)


from sklearn.metrics import accuracy_score
# Print the classification report of `y_test` and `predicted`
print("SVM accuracy: %r" % accuracy_score(predicted, y_test))

# from sklearn import metrics
# print(metrics.classification_report(y_test, predicted))



# SVM3
# # -*- coding: utf-8 -*-
# # @Time    : 2017/7/13 下午8:23
# # @Author  : play4fun
# # @File    : 47.2-使用SVM进行-手写数据OCR.py
# # @Software: PyCharm

# """
# 47.2-使用SVM进行-手写数据OCR.py:
# """

# import cv2
# import numpy as np

# SZ = 20
# bin_n = 16  # Number of bins
# affine_flags = cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR


# # 使用方向梯度直方图Histogram of Oriented Gradients  HOG 作为特征向量
# def deskew(img):
#     m = cv2.moments(img)
#     if abs(m['mu02']) < 1e-2:
#         return img.copy()
#     skew = m['mu11'] / m['mu02']
#     M = np.float32([[1, skew, -0.5 * SZ * skew], [0, 1, 0]])
#     img = cv2.warpAffine(img, M, (SZ, SZ), flags=affine_flags)
#     return img


# # 计算图像 X 方向和 Y 方向的 Sobel 导数
# def hog(img):
#     gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
#     gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
#     mag, ang = cv2.cartToPolar(gx, gy)
#     bins = np.int32(bin_n * ang / (2 * np.pi))  # quantizing binvalues in (0...16)
#     bin_cells = bins[:10, :10], bins[10:, :10], bins[:10, 10:], bins[10:, 10:]
#     mag_cells = mag[:10, :10], mag[10:, :10], mag[:10, 10:], mag[10:, 10:]
#     hists = [np.bincount(b.ravel(), m.ravel(), bin_n) for b, m in zip(bin_cells, mag_cells)]
#     hist = np.hstack(hists)  # hist is a 64 bit vector
#     return hist


# # 最后 和前 一样 我们将大图分割成小图。使用每个数字的前 250 个作 为训练数据
# #  后 250 个作为测试数据
# img = cv2.imread('../data/digits.png', 0)

# cells = [np.hsplit(row, 100) for row in np.vsplit(img, 50)]
# # First half is trainData, remaining is testData
# train_cells = [i[:50] for i in cells] #训练数据
# test_cells = [i[50:] for i in cells] #测试数据

# deskewed = [map(deskew, row) for row in train_cells]
# # deskewed = [deskew(row) for row in train_cells]
# # deskewed = map(deskew, train_cells)
# hogdata = [map(hog, row) for row in deskewed]
# # hogdata = [hog(row) for row in deskewed]
# # hogdata = map(hog, deskewed)

# trainData = np.float32(hogdata).reshape(-1, 64)
# responses = np.float32(np.repeat(np.arange(10), 250)[:, np.newaxis])

# svm = cv2.ml.SVM_create()
# svm.setKernel(cv2.ml.SVM_LINEAR)
# svm.setType(cv2.ml.SVM_C_SVC)
# svm.setC(2.67)
# svm.setGamma(5.383)
# svm.train(trainData, cv2.ml.ROW_SAMPLE, responses)
# svm.save('svm_data.dat')

# deskewed = [map(deskew, row) for row in test_cells]
# hogdata = [map(hog, row) for row in deskewed]
# testData = np.float32(hogdata).reshape(-1, bin_n * 4)

# result = svm.predict(testData)
# mask = result == responses
# correct = np.count_nonzero(mask)
# print(correct * 100.0 / result.size)
# # 94%