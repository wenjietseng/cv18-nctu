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

        # blending here
        # self.img1, self.result_img = self.pyramid_blending(self.img1, self.result_img, transform_dist)
        # original code directly paste two imgs together
        self.result_img[transform_dist[1]:w1+transform_dist[1], 
                    transform_dist[0]:h1+transform_dist[0]] = self.img1


        # cv2.imshow('image', out)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()


    def alpha_blend(self, img_f, img_b, transform_dist, fix_width=100): 
        # blend the overlapping area of two image would be the simplest way
        h1, w1 = img_f.shape[:2]

        foreground = img_f[ :, w1-fix_width:w1,]
        background = img_b[ transform_dist[0]:h1+transform_dist[0], w1+transform_dist[1]:w1+transform_dist[1]+fix_width,]
        # print(w1, h1)
        # print(foreground.shape, background.shape)
        kernel = np.ones((5, 5), np.uint8)
        foreground_gray = cv2.cvtColor(foreground, cv2.COLOR_BGR2GRAY)
        ret, mask = cv2.threshold(foreground_gray, 240, 255, cv2.THRESH_BINARY)

        opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        output = np.zeros(foreground.shape, dtype=foreground.dtype)

        for i in range(3):
            output[:, :, i] = background[:, :, i] * (opening/255) + foreground[:, :, i] *(1-opening/255)
        img_f[:, w1-fix_width:w1] = output
        # img_b[transform_dist[0]:h1+transform_dist[0], w1+transform_dist[1]:w1+transform_dist[1]+fix_width] = output
        return img_f

    def pyramid_blending(self, img_f, img_b, transform_dist, fix_width=100):
        # A is forground, B is background
        # generate Gaussian pyramid for A
        h1, w1 = img_f.shape[:2]

        A = img_f[ :, w1-fix_width:w1,]
        B = img_b[ transform_dist[0]:h1+transform_dist[0], w1+transform_dist[1]:w1+transform_dist[1]+fix_width,]
        G = A.copy()
        gpA = [G]
        for i in range(6):
            G = cv2.pyrDown(G)
            gpA.append(G)
        # generate Gaussian pyramid for B
        G = B.copy()
        gpB = [G]
        for i in range(6):
            G = cv2.pyrDown(G)
            gpB.append(G)
        # generate Laplacian Pyramid for A
        lpA = [gpA[5]]
        for i in range(5,0,-1):
            GE = cv2.pyrUp(gpA[i])
            L = cv2.subtact(gpA[i-1],GE)
            lpA.append(L)

        # generate Laplacian Pyramid for B
        lpB = [gpB[5]]
        for i in range(5,0,-1):
            GE = cv2.pyrUp(gpB[i])
            L = cv2.subtract(gpB[i-1],GE)
            lpB.append(L)

        # Now add left and right halves of images in each level
        LS = []
        for la,lb in zip(lpA,lpB):
            rows,cols,dpt = la.shape
            ls = np.hstack((la[:,0:cols/2], lb[:,cols/2:]))
            LS.append(ls)

        # now reconstruct
        ls_ = LS[0]
        for i in range(1,6):
            ls_ = cv2.pyrUp(ls_)
            ls_ = cv2.add(ls_, LS[i])
        # image with direct connecting each half
        real = np.hstack((A[:,:cols/2],B[:,cols/2:]))
        A[:, w1-fix_width:w1] = ls_
        B[transform_dist[0]:h1+transform_dist[0], w1+transform_dist[1]:w1+transform_dist[1]+fix_width] = ls_
        return A, B
