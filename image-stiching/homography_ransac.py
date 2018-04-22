import numpy as np
import cv2

def find_homography(good, kp1, kp2):
    if len(good) > 4:

        src_pts = np.float32([kp1[m.queryIdx].pt for m in good])\
            .reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good])\
            .reshape(-1, 1, 2)

        H, s = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 4)
        return H
    return None