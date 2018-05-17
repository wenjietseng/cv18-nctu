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
import scipy.cluster.vq
from sklearn.metrics import confusion_matrix
from plot_confusion_matrix import plot_confusion_matrix

CLUSTERNUMBER = 20

# Function used to read data
def dataRead():
    # Path to train data
    train_path = "./hw4_data/train/"
    train_img_list = []

    # Read the folder names
    img_dirs = [d for d in os.listdir(train_path) if not d.startswith('.')]

    # Read all the file names in folders, one folder by one folder
    img_names = [os.listdir(train_path + d) for d in img_dirs if not d.startswith('.')]

    # Read all the images, and record their directories
    for d_idx in range(len(img_dirs)):
        for name in img_names[d_idx]:
            if not name.startswith('.'):
                img = cv2.imread(train_path + img_dirs[d_idx] + '/' + name, 0)
                train_img_list.append((img_dirs[d_idx], img))
   
    # Print out the number of training images
    print("Number of training images: %d " % len(train_img_list))

    # Path to test data
    test_path = "./hw4_data/test/"
    test_img_list = []

    # Read all the file names in folders, one folder by one folder
    img_names = [os.listdir(test_path + d) for d in img_dirs if not d.startswith('.')]

    # Read all the images, and record their directories
    for d_idx in range(len(img_dirs)):
        for name in img_names[d_idx]:
            if not name.startswith('.'):
                img = cv2.imread(test_path + img_dirs[d_idx] + '/' + name, 0)
                test_img_list.append((img_dirs[d_idx], img))
    
    # Print out the number of training images
    print("Number of testing images: %d " % len(test_img_list))

    # Return data
    return img_dirs, train_img_list, test_img_list

# Generate bags of SIFT descriptors
def bagOfSIFT(train_img_list, test_img_list):
    # Create extractor
    extractor = cv2.xfeatures2d.SIFT_create()

    # TRAIN IMAGE
    # Compute descriptors, and store into descriptor array
    train_descriptors_list = []
    for img_idx in range(len(train_img_list)):
        _, descriptors = extractor.detectAndCompute(train_img_list[img_idx][1], None)
        train_descriptors_list.append((train_img_list[img_idx][0], descriptors))

    # Transform the descriptor list into numpy array
    "Stored as an N*128 descriptor array, where N = number of all KPs in ALL IMAGES"
    # Initiate array with first descriptor
    train_descriptors = train_descriptors_list[0][1]
    for _ , image_descriptor in train_descriptors_list[1: ]:
        train_descriptors = np.vstack((train_descriptors, image_descriptor))

    # Do k-means clustering
    # Generate train images' codebook
    train_codebook, _ = scipy.cluster.vq.kmeans(train_descriptors, CLUSTERNUMBER, 1)

    # Assign descriptors to centroids and calculate the histogram
    train_hist = np.zeros((len(train_img_list), CLUSTERNUMBER), "float32")
    for img_idx in range(len(train_img_list)):

        # Assign the descriptors by images
        mapping, _ = scipy.cluster.vq.vq(train_descriptors_list[img_idx][1], train_codebook)

        # Review the result of mapping of descriptors from this image, and record the point
        # accumulation in each centroid
        for mapped_label in mapping:
            train_hist[img_idx][mapped_label] += 1

    # TEST IMAGE
    # Compute descriptors, and store into descriptor array
    test_descriptors_list = []
    for img_idx in range(len(test_img_list)):
        _, descriptors = extractor.detectAndCompute(test_img_list[img_idx][1], None)
        test_descriptors_list.append((test_img_list[img_idx][0], descriptors))

    # Transform the descriptor list into numpy array
    "Stored as an N*128 descriptor array, where N = number of all KPs in ALL IMAGES"
    # Initiate array with first descriptor
    test_descriptors = test_descriptors_list[0][1]
    for _ , image_descriptor in test_descriptors_list[1: ]:
        test_descriptors = np.vstack((test_descriptors, image_descriptor))

    # Do k-means clustering
    # Generate test images' codebook
    test_codebook, _ = scipy.cluster.vq.kmeans(test_descriptors, CLUSTERNUMBER, 1)

    # Assign descriptors to centroids and calculate the histogram
    test_hist = np.zeros((len(test_img_list), CLUSTERNUMBER), "float32")
    for img_idx in range(len(test_img_list)):

        # Assign the descriptors by images
        mapping, _ = scipy.cluster.vq.vq(test_descriptors_list[img_idx][1], test_codebook)

        # Review the result of mapping of descriptors from this image, and record the point
        # accumulation in each centroid
        for mapped_label in mapping:
            test_hist[img_idx][mapped_label] += 1

    # Return clustered result of every image in form of histogram
    return train_hist, test_hist

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

# Read in all data
img_dirs, train_img_list, test_img_list = dataRead()

# Find descriptors and cluster. Returned in np array form
train_hist, test_hist = bagOfSIFT(train_img_list, test_img_list)

# print(train_hist.shape)
# print(test_hist.shape)

# Create lable dictionary
label_dict = {}
for idx, label in enumerate(img_dirs):
     label_dict[idx] = label

# Create Y results
train_Y = np.repeat(np.arange(15), 100)
test_Y = np.repeat(np.arange(15), 10)

nn = NearestNeighbor()
nn.train(train_hist, train_Y)
Yte_predict_L1 = nn.predict(test_hist, 'L1')
print('Bag of SIFT, NN with L1 norm: %f' % (np.mean(Yte_predict_L1 == test_Y)))
Yte_predict_L2 = nn.predict(test_hist, 'L2')
print('Bag of SIFT, NN with L2 norm: %f' % (np.mean(Yte_predict_L2 == test_Y)))

# Confusion matrix
con_mat_L1 = confusion_matrix(test_Y, Yte_predict_L1)
con_mat_L2 = confusion_matrix(test_Y, Yte_predict_L2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(con_mat_L1, classes=img_dirs,
                      title='Confusion matrix, NN')
plt.savefig('./task_2_out/confustion_mat_nn.png', bbox_inches='tight', dpi=300)
plt.close()


# Show the histogram
# print(len(train_hist))
# print(len(test_hist))