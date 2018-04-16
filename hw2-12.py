import numpy as np
import cv2
import sys
from scipy.spatial.distance import cdist
import sys, math, skimage.io as io, matplotlib.pyplot as plt, numpy as np



"""
my previous script
        src_pts = np.float32([self.kp1[m.queryIdx].pt for m in good_matches])\
            .reshape(-1, 1, 2)
        dst_pts = np.float32([self.kp2[m.trainIdx].pt for m in good_matches])\
            .reshape(-1, 1, 2)

        print(src_pts)
        print(dst_pts)
        if len(src_pts) > 4:
            self.M, self.mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5)
        else:
            self.M = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
"""




# FAST keypoint detection
def fastKPDetection(img, numberOfImage):

    # Create a fast detector for keypoint detection
    detector = cv2.cv2.FastFeatureDetector_create()

    # Detect keypoint
    keyPointData = detector.detect(img)
    result = cv2.cv2.drawKeypoints(img, keyPointData, outImage = np.array([]), color=(255,0 ,0))

    # Save the keypoint image
    cv2.cv2.imwrite("test_" + str(numberOfImage) + ".png", result)

    # Print out the number of keypoints
    print("# of keypoints: {}".format(len(keyPointData)))

    # return the keypoint
    return keyPointData

# SIFT feature extractor, root SIFT included to enable more precise result
def SIFTFeatureExtractor(masterImg, slaveImg, numberOfImage):

    # Create SIFT extractor
    extractor = cv2.cv2.xfeatures2d.SIFT_create()

    # Compute feature vectors, and store into featureVectors array
    keyPoint_0, featureVectors_0 = extractor.detectAndCompute(masterImg, None)
    keyPoint_1, featureVectors_1 = extractor.detectAndCompute(slaveImg, None)

    # Draw the keypoint images
    result_0 = cv2.cv2.drawKeypoints(masterImg, keyPoint_0, outImage = np.array([]), color=(255,0 ,0))
    result_1 = cv2.cv2.drawKeypoints(slaveImg, keyPoint_1, outImage = np.array([]), color=(255,0 ,0))

    # Store the keypoint images
    cv2.cv2.imwrite("kpImage_" + str(numberOfImage) + ".png", result_0)
    cv2.cv2.imwrite("kpImage_" + str(numberOfImage + 1) + ".png", result_1)

    # If keypoint detection succeeded
    if(len(keyPoint_0) > 0) and (len(keyPoint_1) > 0):

        # Print out information
        print("[INFO] # of keypoints detected in master image: {}".format(len(keyPoint_0)))
        print("[INFO] feature vector shape in master image: {}".format(featureVectors_0.shape))
        print("[INFO] # of keypoints detected in slave image: {}".format(len(keyPoint_1)))
        print("[INFO] feature vector shape in slave image: {}".format(featureVectors_1.shape))

        # Retrun extracted data
        return (keyPoint_0, featureVectors_0, keyPoint_1, featureVectors_1)
    
    else:
        return ([], None, [], None)

# Feature matching function
def featureMatching(masterFeature, slaveFeature):

    # Create a bruteforce matcher
    matcher = cv2.cv2.BFMatcher()

    # Roughly match closest features into pairs
    rawMatches = matcher.knnMatch(masterFeature, slaveFeature, 2)

    # Create an array for final result
    matches = []

    # David Lowe's ratio test; ratio is recommended to set between 0.7 and 0.8
    testRatio = 0.7
    for currentMatch in rawMatches:
        if len(currentMatch) == 2 and currentMatch[0].distance < currentMatch[1].distance * testRatio:
            matches.append([currentMatch[0]])

    # Return the matches left after the test
    return matches

# RANSAC algorithm to get best matching pair
def ransac(data, tolerance=0.5, max1=100, confidence=0.95):
    count, bm, bc, bi = 0, None, 0, None
    while count < max1:
        tempd, temps = np.matrix(np.copy(data)), np.copy(data)
        np.random.shuffle(temps) # Gets a random set of points based on RANSAC
        temps = np.matrix(temps)[0:4]
        homography = getHomography(temps[:,0:2], temps[:,2:])
        error = np.sqrt((np.array(np.array(homogeneous((homography * homogeneous(tempd[:,0:2].transpose())))[0:2,:]) - tempd[:,2:].transpose()) ** 2).sum(0))
        if (error < tolerance).sum() > bc:
            bm, bc, bi = homography, (error < tolerance).sum(), np.argwhere(error < tolerance)
            p = float(bc) / data.shape[0]
            max1 = math.log(1-confidence)/math.log(1-(p**4))
        count += 1
    return bm, bi

# Converts 3 x N set of points to homogenous coordinates
def homogeneous(fir):
    if fir.shape[0] == 3:
        out = np.zeros_like(fir)
        for i in range(3):
            out[i, :] = fir[i, :] / fir[2, :]
    elif fir.shape[0] == 2: out = np.vstack((fir, np.ones((1, fir.shape[1]), dtype=fir.dtype)))
    return out

# Gets the homography on an image. Based on 4 point coordinate system.
def getHomography(p1, p2):
    A = np.matrix(np.zeros((p1.shape[0]*2, 8), dtype=float), dtype=float)
    # Filling out A based on equation online
    for i in range(0, A.shape[0]):
        if i % 2 == 0:
            A[i,0], A[i,1], A[i,2], A[i,6], A[i,7] = p1[i/2,0], p1[i/2,1], 1, -p2[i/2,0] * p1[i/2,0], -p2[i/2,0] * p1[i/2,1]
        else:
            A[i,3], A[i,4], A[i,5], A[i,6], A[i,7] = p1[i/2,0], p1[i/2,1], 1, -p2[i/2,1] * p1[i/2,0], -p2[i/2,1] * p1[i/2,1]

    # Creating b based on equation
    b = p2.flatten().reshape(p2.flatten().shape[1], 1).astype(float)
    
    # Calculating homography A * x = b
    x = np.linalg.solve(A,b) if p1.shape[0] == 4 else np.linalg.lstsq(A,b)[0]
    return np.vstack((x, np.matrix(1))).reshape((3,3))



# Set parameters
numberOfImage = 0

# Read image
img_0 = cv2.cv2.imread(sys.argv[1], cv2.cv2.IMREAD_GRAYSCALE)
img_1 = cv2.cv2.imread(sys.argv[2], cv2.cv2.IMREAD_GRAYSCALE)

# Extract feature using SIFT
(keyPoint_0, featureVectors_0, keyPoint_1, featureVectors_1) = SIFTFeatureExtractor(img_0, img_1, numberOfImage)



# Generate the matching image
img_final = cv2.cv2.drawMatchesKnn(img_0, keyPoint_0, img_1, keyPoint_1, matches, None, flags = 2)
cv2.cv2.imwrite("final.png", img_final)

# Main Code
# Getting homography of image files. Supports two for now.
print("Creating the panorama!")
hh, images, midh, wl = [np.matrix(np.identity(3))], [sys.argv[1], sys.argv[2]], [], []
for i in range(len(images) - 1):
    image1b, image2b = img_0, img_1
    points1, points2 = harris(image1b, count=500), harris(image2b, count=500)
    # displayPoints(io.imread(images[i]), points1)
    # modified = anms(points1, top=150)
    # displayPoints(io.imread(images[i]), modified)
    # Match the features
    matches = featureMatching(featureVectors_0, featureVectors_1)
    h = ransac(np.matrix(np.hstack((points1[matches[:,0],0:2], points2[matches[:,1],0:2]))),0.5)
    hh.append(np.linalg.inv(h[0]))



# import numpy as np
# import cv2

# img1 = cv2.imread("box.png",0)          # queryImage
# img2 = cv2.imread("box_in_scene.png",0) # trainImage

# # Initiate SIFT detector

# sift = cv2.xfeatures2d.SIFT_create()
# # find the keypoints and descriptors with SIFT
# kp1, des1 = sift.detectAndCompute(img1,None)
# kp2, des2 = sift.detectAndCompute(img2,None)

# # BFMatcher with default params
# bf = cv2.BFMatcher()
# matches = bf.knnMatch(des1,des2, k=2)

# # Apply ratio test
# good = []
# for m,n in matches:
#     if m.distance < 0.75*n.distance:
#         good.append([m])

# # cv2.drawMatchesKnn expects list of lists as matches.
# img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good,None, flags=2)
# cv2.cv2.imwrite("final.png", img3)