import cv2, sys
import numpy as np
# from matplotlib import pyplot as plt
# from PIL import Image

class interest_point_detection(object):
    def __init__(self, img_path):
        self.img_path = img_path

        self.read_image()
        # self.show_img()
    
        self.kp_features_sift()
        # self.show_img()
        self.write_img()
    
    def read_image(self):
        # read image and turn it into gray scale, consider 3 color channels?
        # with cv2.imread(self.img_path) as img:
        self.img = cv2.imread(self.img_path)
        self.img_gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)

    def kp_features_sift(self):
        self.sift = cv2.xfeatures2d.SIFT_create()
        self.interest_point, self.descriptors = self.sift.detectAndCompute(self.img_gray, None)
        self.img = cv2.drawKeypoints(self.img_gray, self.interest_point, self.img)

    # def kp_features_mser(self):
        # self.mser = cv2.MSER_create()


    def show_img(self):
        cv2.imshow('image', self.img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def write_img(self):
        cv2.imwrite('out.jpg', self.img)



if __name__ == "__main__":
    fname = sys.argv[1]
    interest_point_detection(fname)
# img=cv2.drawKeypoints(gray,kp,img,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
# cv2.imwrite('sift_keypoints.jpg',img)
# kp, des = sift.detectAndCompute(gray,None)