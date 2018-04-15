import cv2
import numpy as np
from matplotlib import pyplot as plt
import sys

from interest_point_detect import interest_point_detect
from feature_matching import feature_matching
from homography import homography
from panoramic_image_stiching import image_stiching

# main script

fname1 = 'IMG_4477.JPG'
fname2 = 'IMG_4479.JPG'

img_a = interest_point_detect(fname1)
img_b = interest_point_detect(fname2)
match_test = feature_matching(img_a.kp, img_b.kp, img_a.des, img_b.des, img_a.img_gray, img_b.img_gray)
