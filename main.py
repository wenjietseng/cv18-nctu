import cv2
import numpy as np
import math
import multiprocessing as mp
import sys
# load scripts
import utils
import feature_match
import homography_ransac
import pano_stich


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print('Please enter img dir!')
        print('python3 main.py ./path_to_img_dir')
        sys.exit(0)
    input_dir = sys.argv[1]

    pool = mp.Pool(mp.cpu_count())
    print(pool)

    img_list, focal_length = utils.parse(input_dir)

    print('==> cylinder projection')
    cylinder_img_list = pool.starmap(utils.cylindrical_projection, [(img_list[i], focal_length[i]) for i in range(len(img_list))])

    _, img_width, _  = img_list[0].shape
    first_stitched_img = cylinder_img_list[0].copy()

    shifts = [[0, 0]]
    
    for i in range(1, len(cylinder_img_list)):
        print("We are at " + str(i) + " and " + str(i+1) + " images. " + str(len(cylinder_img_list)) + "in total.")
        img1 = cylinder_img_list[i - 1]
        img2 = cylinder_img_list[i]

        print("--- Feature matching ---", end='', flush=True)
        f_match = feature_match.matcher(img1, img2)
        matches = f_match.matches
        kp1 = f_match.kp1
        kp2 = f_match.kp2

        
#find_H = homography(f_match.matches, f_match.kp1, f_match.kp2, f_match.img1)
#print(find_H.H)
#stich = image_stiching(f_match.img1, f_match.img2, find_H.H)
#cv2.imwrite('out-final.png', stich.result_img)
