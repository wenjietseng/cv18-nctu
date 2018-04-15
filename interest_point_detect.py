import cv2, sys
import numpy as np
from matplotlib import pyplot as plt

class interest_point_detect(object):
    def __init__(self, img_path):
        self.img_path = img_path
        self.read_image()
        # self.show_img(self.img)
        # self.show_img(self.img_gray)
        self.kp_features_sift()
        # self.show_img(self.img)
        # self.show_img(self.img_gray)
        self.write_img(self.img_gray)
    
    def read_image(self):
        self.img = cv2.imread(self.img_path)
        if self.img is None:
            print('Invalid image:' + self.img_path)
        else:
            print('Image successfully read ...')
        # resize for easier visualize
        self.img = cv2.resize(self.img, (400, 400), interpolation = cv2.INTER_AREA)
        self.img_gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY) # or specify 0 in imread

    def kp_features_sift(self):
        """ 
        We use gray scale image and SIFT to compute
            kp: key points or interest points
            des: descriptors
        """
        self.sift = cv2.xfeatures2d.SIFT_create()
        self.kp, self.des = self.sift.detectAndCompute(self.img_gray, None)
        self.img_gray = cv2.drawKeypoints(self.img_gray, self.kp, self.img_gray)

    def kp_features_mser(self):
        pass

    def show_img(self, img):
        cv2.imshow('image', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def write_img(self, img):
        cv2.imwrite('out-interest_point.jpg', img)