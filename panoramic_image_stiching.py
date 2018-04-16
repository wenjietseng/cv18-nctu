import cv2, sys
import numpy as np
from matplotlib import pyplot as plt

class image_stiching(object):
    def __init__(self, img1, img2, H):
        self.img1 = img1
        self.img2 = img2
        self.H = H
        self.wrap_image()
    
    def wrap_image(self):
        # Get width and height of input images	
        w1, h1 = self.img1.shape[:2]
        w2, h2 = self.img2.shape[:2]
        img1_dims = np.float32([ [0,0], [0,w1], [h1, w1], [h1,0] ]).reshape(-1,1,2)
        img2_dims_temp = np.float32([ [0,0], [0,w2], [h2, w2], [h2,0] ]).reshape(-1,1,2)

        # Get relative perspective of second image
        img2_dims = cv2.perspectiveTransform(img2_dims_temp, self.H)

        # Resulting dimensions
        result_dims = np.concatenate( (img1_dims, img2_dims), axis = 0)

        # Getting images together
        # Calculate dimensions of match points
        [x_min, y_min] = np.int32(result_dims.min(axis=0).ravel() - 0.5)
        [x_max, y_max] = np.int32(result_dims.max(axis=0).ravel() + 0.5)

        # Create output array after affine transformation 
        transform_dist = [-x_min,-y_min]
        transform_array = np.array([[1, 0, transform_dist[0]], 
                                    [0, 1, transform_dist[1]], 
                                    [0,0,1]]) 

        # Warp images to get the resulting image
        self.result_img = cv2.warpPerspective(self.img2, transform_array.dot(self.H), 
                                    (x_max-x_min, y_max-y_min))
        self.result_img[transform_dist[1]:w1+transform_dist[1], 
                    transform_dist[0]:h1+transform_dist[0]] = self.img1

