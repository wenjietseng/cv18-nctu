import cv2
import numpy as np
import sys

A = cv2.imread(sys.argv[1])

# generate Gaussian pyramid for A
G = A.copy()
gpA = [G]
for i in range(6): #範圍從0到6(下面range裡訂的範圍，用10好像會有OutOfMemoryError)
    G = cv2.pyrDown(G) #pyrDown:製作Gaussian pyramid
    gpA.append(G)

# construct
ls_ = G
for i in range(1,6): #訂定範圍
    ls_ = cv2.pyrUp(ls_)

cv2.imwrite('1-2_Pyramid\Gaussian Pyramids_motorcycle\Pyramid_6.jpg', ls_) 