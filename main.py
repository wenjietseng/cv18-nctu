import cv2
import numpy as np
from matplotlib import pyplot as plt
import sys

import feature_match
import homography_ransac
import panoramic_image_stiching


# main script
def main():
    fname1 = './photoSource/lab/IMG_0023.JPG'
    fname2 = './photoSource/lab/IMG_0024.JPG'

    # find matches
    f_matches = feature_match.matcher(fname1, fname2)

    # use matched point pairs and ransac to find homography matrix
    H = homography_ransac.find_homography(f_matches.matches, f_matches.kp1, f_matches.kp2)

    # stich image using homography
    stich = panoramic_image_stiching.image_stiching(f_matches.img1, f_matches.img2, H)

    cv2.imwrite('out-final.png', stich.result_img)

    # crop
    crop_img = crop(stich.result_img)

    cv2.imwrite('out-final-cropped.png', crop_img)

def crop(img):
    """Crop the black border in image
    Args:
        img: a panoramas image
    Returns:
        Cropped image
    """
    _, thresh = cv2.threshold(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 1, 255, cv2.THRESH_BINARY)
    upper, lower = [-1, -1]
    black_pixel_num_threshold = img.shape[1]//10

    for y in range(thresh.shape[0]):
        if len(np.where(thresh[y] == 0)[0]) < black_pixel_num_threshold:
            upper = y
            break
        
    for y in range(thresh.shape[0]-1, 0, -1):
        if len(np.where(thresh[y] == 0)[0]) < black_pixel_num_threshold:
            lower = y
            break

    return img[upper:lower, :]

if __name__ == '__main__':
    main()