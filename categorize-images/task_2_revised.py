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
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np
import cv2, os
import matplotlib.pyplot as plt
import scipy.cluster.vq

CLUSTERNUMBER = 50

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
    for d_idx in range(5):
        for name in img_names[d_idx]:
            if not name.startswith('.'):
                img = cv2.imread(train_path + img_dirs[d_idx] + '/' + name, 0)
                # img = cv2.resize(img, (200,200), interpolation=cv2.INTER_CUBIC)
                train_img_list.append((img_dirs[d_idx], img))
   
    # Print out the number of training images
    print("Number of training images: %d " % len(train_img_list))

    # Path to test data
    test_path = "./hw4_data/test/"
    test_img_list = []

    # Read all the file names in folders, one folder by one folder
    img_names = [os.listdir(test_path + d) for d in img_dirs if not d.startswith('.')]

    # Read all the images, and record their directories
    for d_idx in range(5):
        for name in img_names[d_idx]:
            if not name.startswith('.'):
                img = cv2.imread(test_path + img_dirs[d_idx] + '/' + name, 0)
                # img = cv2.resize(img, (200,200), interpolation=cv2.INTER_CUBIC)
                test_img_list.append((img_dirs[d_idx], img))
    
    # Print out the number of training images
    print("Number of testing images: %d " % len(test_img_list))

    # Return data
    return img_dirs, train_img_list, test_img_list

# Generate bags of SIFT descriptors
def bagOfSIFT(train_img_list):
    # Create extractor
    extractor = cv2.xfeatures2d.SIFT_create()

    # TRAIN IMAGE
    # Compute descriptors, and store into descriptor array
    train_descriptors_list = []
    for img_idx in range(len(train_img_list)):
        _, descriptors = extractor.detectAndCompute(train_img_list[img_idx][1], None)
        train_descriptors_list.append((train_img_list[img_idx][0], descriptors))

    # Transform the descriptor list into numpy array
    # Stored as an N*128 descriptor array, where N = number of all KPs in ALL IMAGES
    # Initiate array with first descriptor
    train_descriptors = train_descriptors_list[0][1]
    for _ , image_descriptor in train_descriptors_list[1: ]:
        train_descriptors = np.vstack((train_descriptors, image_descriptor))

    print('train descriptor shape: ')
    print(train_descriptors.shape)

    # Do k-means clustering
    # Generate train images' codebook
    # train_codebook, _ = scipy.cluster.vq.kmeans(train_descriptors, CLUSTERNUMBER, 1)

    num_clusters = 50
    num_imgs = 500

    kmeans_obj = KMeans(n_clusters=num_clusters)
    kmeans_ret = kmeans_obj.fit_predict(train_descriptors)
    # print(kmeans_ret.shape)
    # print(kmeans_ret) # Index of the cluster each sample belongs to.

    mega_histogram = np.array([np.zeros(num_clusters) for i in range(num_imgs)])
    old_count = 0
    for i in range(num_imgs):
        l = len(train_descriptors_list[i])
        for j in range(l):
            idx = kmeans_ret[old_count+j]
            mega_histogram[i][idx] += 1
        old_count += 1
    # print(mega_histogram)

    # print(np.sum(mega_histogram))
    # print(np.sum(mega_histogram, axis=0))
    # print(np.argmax(mega_histogram))

    print('vstack finish, do scaling')
    scale = StandardScaler().fit(mega_histogram)
    mega_histogram = scale.transform(mega_histogram)
    print(mega_histogram.shape)
    print('finish train feature')
    
    # TEST IMAGE
    # Compute descriptors, and store into descriptor array
    # test_descriptors_list = []
    # for img_idx in range(len(test_img_list)):
    #     _, descriptors = extractor.detectAndCompute(test_img_list[img_idx][1], None)
    #     test_descriptors_list.append((test_img_list[img_idx][0], descriptors))

    # Transform the descriptor list into numpy array
    # Stored as an N*128 descriptor array, where N = number of all KPs in ALL IMAGES
    # Initiate array with first descriptor
    # test_descriptors = test_descriptors_list[0][1]
    # for _ , image_descriptor in test_descriptors_list[1: ]:
    #     test_descriptors = np.vstack((test_descriptors, image_descriptor))

    # test_ret = kmeans_obj.predict(test_descriptors)


    # test_histogram = np.array([np.zeros(num_clusters) for i in range(50)])
    # old_count2 = 0
    # for i in range(50):
    #     l = len(test_descriptors_list[i])
    #     for j in range(l):
    #         idx = test_ret[old_count2+j]
    #         test_histogram[i][idx] += 1
    #     old_count2 += 1
    # # print(test_histogram)
    # test_histogram = scale.transform(test_histogram)
    # # print(test_histogram)
    return mega_histogram, kmeans_obj
    # return train_hist, test_hist

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
def recognize(img, km_obj):

    """ 
    This method recognizes a single image 
    It can be utilized individually as well.
    """
    extractor = cv2.xfeatures2d.SIFT_create()
    _, des = extractor.detectAndCompute(img, None)
    kmeans_obj = km_obj
    # print kp
    # print(des.shape)

    # generate vocab for test image
    vocab = np.array( [[ 0 for i in range(CLUSTERNUMBER)]])
    # locate nearest clusters for each of 
    # the visual word (feature) present in the image
    
    # test_ret =<> return of kmeans nearest clusters for N features
    test_ret = kmeans_obj.predict(des)
    # print test_ret

    # print vocab
    for each in test_ret:
        vocab[0][each] += 1

    # print(vocab)

    scale = StandardScaler().fit(vocab)
    # Scale the features
    vocab = scale.transform(vocab)

    # predict the class of the image
    # lb = self.bov_helper.clf.predict(vocab)
    # print "Image belongs to class : ", self.name_dict[str(int(lb[0]))]
    return vocab 


# Read in all data
img_dirs, train_img_list, test_img_list = dataRead()

# Find descriptors and cluster. Returned in np array form
train_hist, km_obj = bagOfSIFT(train_img_list)

y_pred = []
for i in test_img_list:
    img = i[1]
    v = recognize(img, km_obj)
    y_pred.append(v)
y_pred = np.asarray(y_pred).reshape((50, 50))
print(y_pred.shape)

# Create lable dictionary
label_dict = {}
for idx, label in enumerate(img_dirs):
     label_dict[idx] = label

# Create Y results
train_Y = np.repeat(np.arange(5), 100)
test_Y = np.repeat(np.arange(5), 10)

nn = NearestNeighbor()

print('before training')
print(train_hist.shape)
print(train_Y.shape)
nn.train(train_hist, train_Y)
Yte_predict_L1 = nn.predict(y_pred, 'L1')
print('Bag of SIFT, NN with L1 norm: %f' % (np.mean(Yte_predict_L1 == test_Y)))
Yte_predict_L2 = nn.predict(y_pred, 'L2')
print('Bag of SIFT, NN with L2 norm: %f' % (np.mean(Yte_predict_L2 == test_Y)))

print(Yte_predict_L1)

# Show the histogram
# print(len(train_hist))
# print(len(test_hist))