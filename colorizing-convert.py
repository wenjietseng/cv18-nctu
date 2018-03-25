# divide raw image into three images with same size
import numpy as np
import cv2
import sys



img = cv2.imread("./hw1_data/task3_colorizing/cathedral.jpg")
# cv2.imshow('image', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# create an array with rgb 3 dimensions,
# data type is unit8 since the value ranges from 0 to 255

print(img.shape)
print(len(img))
print(len(img[0]))
