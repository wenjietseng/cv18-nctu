import cv2
import sys
import numpy as np

class matcher(object):
    def __init__(self, img_path1, img_path2):
        self.img1 = cv2.imread(img_path1, cv2.IMREAD_GRAYSCALE)
        self.img2 = cv2.imread(img_path2, cv2.IMREAD_GRAYSCALE) 
        
        # a simple resize to avoid image grow too big
        h1, w1 = self.img1.shape[:2]
        self.img1 = cv2.resize(self.img1, (int(1/8*w1), int(1/8*h1)), interpolation=cv2.INTER_CUBIC)
        h2, w2 = self.img2.shape[:2]
        self.img2 = cv2.resize(self.img2, (int(1/8*w2), int(1/8*h2)), interpolation=cv2.INTER_CUBIC) 
        
        # Set parameters
        self.numberOfImage = 0

        # Extract feature using SIFT
        (self.kp1, self.des1, self.kp2, self.des2) = self.SIFTFeatureExtractor(self.img1, self.img2, self.numberOfImage)
        
        # Match the features
        self.matches, self.matches_for_draw = self.featureMatching(self.des1, self.des2)

        # Draw matches and save
        img_final = cv2.drawMatchesKnn(self.img1, self.kp1, self.img2, self.kp2, self.matches_for_draw, None, flags = 2)
        cv2.imwrite("final.png", img_final)

    
    def SIFTFeatureExtractor(self, masterImg, slaveImg, numberOfImage):
        # SIFT feature extractor, root SIFT included to enable more precise result
        # Create SIFT extractor
        extractor = cv2.xfeatures2d.SIFT_create()
        # Compute feature vectors, and store into featureVectors array
        keyPoint_0, featureVectors_0 = extractor.detectAndCompute(masterImg, None)
        keyPoint_1, featureVectors_1 = extractor.detectAndCompute(slaveImg, None)

        # Draw the keypoint images
        result_0 = cv2.drawKeypoints(masterImg, keyPoint_0, outImage = np.array([]), color=(255,0 ,0))
        result_1 = cv2.drawKeypoints(slaveImg, keyPoint_1, outImage = np.array([]), color=(255,0 ,0))

        # Store the keypoint images
        cv2.imwrite("kpImage_" + str(numberOfImage) + ".png", result_0)
        cv2.imwrite("kpImage_" + str(numberOfImage + 1) + ".png", result_1)

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
    def featureMatching(self, masterFeature, slaveFeature):

        # Create a bruteforce matcher
        matcher = cv2.BFMatcher()

        # Roughly match closest features into pairs
        rawMatches = matcher.knnMatch(masterFeature, slaveFeature, 2)

        # Create an array for final result
        matches = []
        matches_for_draw = []
        # David Lowe's ratio test; ratio is recommended to set between 0.7 and 0.8
        testRatio = 0.7
        for m, n in rawMatches:
            if m.distance < n.distance * testRatio:
                matches.append(m)
                matches_for_draw.append([m])
        # Return the matches left after the test
        return matches, matches_for_draw
