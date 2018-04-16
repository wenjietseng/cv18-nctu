import numpy as np
import cv2
import sys

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

def main():

    # Set parameters
    numberOfImage = 0

    # Read image
    img_0 = cv2.cv2.imread(sys.argv[1], cv2.cv2.IMREAD_GRAYSCALE)
    img_1 = cv2.cv2.imread(sys.argv[2], cv2.cv2.IMREAD_GRAYSCALE)

    # Extract feature using SIFT
    (keyPoint_0, featureVectors_0, keyPoint_1, featureVectors_1) = SIFTFeatureExtractor(img_0, img_1, numberOfImage)

    # Match the features
    matches = featureMatching(featureVectors_0, featureVectors_1)

    # Generate the matching image
    img_final = cv2.cv2.drawMatchesKnn(img_0, keyPoint_0, img_1, keyPoint_1, matches, None, flags = 2)
    cv2.cv2.imwrite("final.png", img_final)

main()
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