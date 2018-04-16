import cv2
import numpy as np
from matplotlib import pyplot as plt
import sys

from interest_point_detect import interest_point_detect
from feature_matching import feature_matching
from homography import homography
from panoramic_image_stiching import image_stiching

# main script

fname1= './photoSource/out-final-0.png'
# fname1 = './photoSource/IMG_0023.JPG'
fname2 = './photoSource/IMG_0025.JPG'
# fname1 = 'img1.png'
# fname2 = 'img4.png'

img_a = interest_point_detect(fname1)
img_b = interest_point_detect(fname2)
f_matches = feature_matching(img_a.kp, img_b.kp, img_a.des, img_b.des, img_a.img_gray, img_b.img_gray)
stich = image_stiching(img_a.img, img_b.img, f_matches.M)
cv2.imwrite('out-final.png', stich.result_img)
