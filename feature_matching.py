import numpy as np
import cv2
from matplotlib import pyplot as plt


class feature_matching(object):
    pass

img1 = cv2.imread('img1.png',0)          # queryImage
img2 = cv2.imread('img4.png',0)          # trainImage

# Initiate SIFT detector
# orb = cv2.ORB()
sift = cv2.xfeatures2d.SIFT_create()
# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)

# print(type(kp1), type(des1), type(kp2), type(des2))
# print(kp1[0].angle)
# print(kp1[0].class_id)
# print(kp1[0].octave)
# print(kp1[0].pt)
# print(kp1[0].response)
# print(kp1[0].size)

# create BFMatcher object
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

# Match descriptors.
print(img1.shape, img2.shape)

matches = bf.match(des1,des2)
# print(matches)
# Sort them in the order of their distance.
matches = sorted(matches, key = lambda x:x.distance)

# Draw first 10 matches.
img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches, None, flags=2)

plt.imshow(img3)
plt.show()


# Brute-Force Matching with SIFT Descriptors and Ratio test