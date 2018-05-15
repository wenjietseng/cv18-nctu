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
import cv2

# Transform the clustered result into histogram
def clusterToHistogram(clt):
    # Create label from 0 to k
    numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)

    # Create histogram
    (hist, _) = np.histogram(clt.labels_, bins = numLabels)

    # Normalize the histogram
    # hist = hist.astype("float")
    # hist /= hist.sum()

    return hist

img = cv2.imread("./hw4_data/train/Bedroom/image_0002.jpg", cv2.IMREAD_GRAYSCALE)

# Create extractor
extractor = cv2.xfeatures2d.SIFT_create()

# Compute feature vectors, and store into featureVectors array
keyPoint_0, descriptors = extractor.detectAndCompute(img, None)
print(descriptors.shape)

# Do k-means clustering
# Initiate KMeans
clt = cluster.KMeans(n_clusters = 4)

# Cluster the histograms
clt.fit(descriptors)

hist = clusterToHistogram(clt)

# Show the histogram
print(hist)
