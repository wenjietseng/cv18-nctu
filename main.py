import cv2
import numpy as np
from matplotlib import pyplot as plt
import sys

from feature_match import matcher
from homography_ransac import homography
from pano_stich import image_stiching

# main script
fname1 = './photoSource/IMG_0023.JPG'
fname2 = './photoSource/IMG_0024.JPG'

f_match = matcher(fname1, fname2)
print('--- img size ---')
print(f_match.img1.shape, f_match.img2.shape)
print('--- matches ---')
print(f_match.matches, len(f_match.matches))
print('--- len of kps ---')
print(len(f_match.kp1), len(f_match.kp2))
print('--- size of descriptors ---')
print(f_match.des1.shape, f_match.des2.shape)
print('--- descriptors ---')
print(f_match.des1, f_match.des2)

homography(f_match.matches, f_match.kp1, f_match.kp2)


# stich = image_stiching(img_a.img, img_b.img, f_matches.M)
# cv2.imwrite('out-final.png', stich.result_img)
