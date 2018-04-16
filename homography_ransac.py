import numpy as np
import cv2
import sys
import math, skimage.io as io, matplotlib.pyplot as plt

class homography(object):
    def __init__(self, good_matches, kp1, kp2):
        self.good_matches, self.kp1, self.kp2 = good_matches, kp1, kp2
        self.homomat()


    def homomat(self):
        src_pts = np.float32([self.kp1[m.queryIdx].pt for m in self.good_matches])\
            .reshape(-1, 1, 2)
        dst_pts = np.float32([self.kp2[m.trainIdx].pt for m in self.good_matches])\
            .reshape(-1, 1, 2)
        print(src_pts)
        print(np.reshape(src_pts, (lensrc)))
        print(np.hstack([dst_pts, src_pts]).shape)


    # RANSAC algorithm to get best matching pair
    def ransac(self, data, tolerance=0.5, max1=100, confidence=0.95):
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

        

"""
my previous script

        if len(src_pts) > 4:
            self.M, self.mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5)
        else:
            self.M = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
"""




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