""" Task 2: Bag of SIFT representation wtih NN
    - use cv2 to find SIFT feature descriptors, which should be Nx128 (N is the number of features detected)
    - Vector Quantization:
        Do K-means clustering to turn descriptors into groups
        historgram the grouping result, code them into another vector
        after this step, apply classifier as we did in task_1.py
    - Nearest neighbor classifier (see task_1.py)
    - can try with KNN + Cross Validation (see task_1.py)
    - Confusion matrix visualization result (see task_1.py)
"""
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
    for d_idx in range(2):
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
    for d_idx in range(2):
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
    clt = cluster.KMeans(n_clusters = 4, random_state = 0)

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
train_Y = np.repeat(np.arange(2), 100)
test_Y = np.repeat(np.arange(2), 10)
    
# Show the histogram
# print(len(train_hist))
# print(len(test_hist))
