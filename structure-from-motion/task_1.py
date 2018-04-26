import cv2
import numpy as np
from matplotlib import pyplot as plt
import random

img1 = cv2.imread('./homework3/Mesona1.JPG',0)  #queryimage # left image
img2 = cv2.imread('./homework3/Mesona2.JPG',0)  #trainimage # right image

K = np.array([[1.4219, 0.0005, 0.5092],
              [0, 1.4219, 0.3802],
              [0, 0, 0.0010]], dtype=float)
K_inv = np.linalg.inv(K)
# convert to normalized coords by pre-multiplying all points with the inverse of calibration matrix
# set first camera's coord to world coord
# X = K_inv * img1
# Xp = K_inv * img2

sift = cv2.xfeatures2d.SIFT_create()

# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

# FLANN parameters
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)

flann = cv2.FlannBasedMatcher(index_params,search_params)
matches = flann.knnMatch(des1,des2,k=2)

good = []
x = []
xp = []

# ratio test as per Lowe's paper
for i, (m, n) in enumerate(matches):
    if m.distance < 0.8 * n.distance:
        good.append(m)
        x.append(kp2[m.trainIdx].pt)
        xp.append(kp1[m.queryIdx].pt)

# fundamental matrix
n = len(x)
if len(xp) != n:
    raise ValueError("Numebr of points don't match.")

# system equation
A = np.zeros((n, 9))

for i in range(n):
    A[i] = [xp[i][1]*x[i][1], xp[i][1]*x[i][0], xp[i][1],
            xp[i][0]*x[i][1], xp[i][0]*x[i][0], xp[i][0],
            x[i][1], x[i][0], 1]

U, S, V = np.linalg.svd(A)
# print(U)
# print('---')
# print(S)
# print('---')
# print(V)

F = V[-1].reshape(3,3)


# resolve det(F) = 0 constraint using SVD
# fundamental matrix has rank 2
U, S, V = np.linalg.svd(F)
# make S into rank 2, zeroing out last singular value
S[2] = 0

F = np.dot(U, np.dot(np.diag(S), V))
print(F/F[2,2])