import numpy as np
import cv2
import sys
import math, skimage.io as io, matplotlib.pyplot as plt
from scipy import *
from scipy import linalg
from scipy import ndimage
from scipy.special import *
from random import choice
from PIL import Image
import warnings
warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")

class homography(object):
    def __init__(self, good_matches, kp1, kp2, img1):
        self.good_matches, self.kp1, self.kp2 = good_matches, kp1, kp2
        if np.ndim(img1) == 2:
            self.rows, self.cols = img1.shape
        else:
            self.rows, self.cols, _ = img1.shape
        self.H = self.homomat()


    def homomat(self):
        src_pts = np.float32([self.kp1[m.queryIdx].pt for m in self.good_matches])\
            .reshape(-1, 1, 2)
        dst_pts = np.float32([self.kp2[m.trainIdx].pt for m in self.good_matches])\
            .reshape(-1, 1, 2)
        
        plist = np.hstack((dst_pts, src_pts))
        H = self.ransac(plist)
        return H

    def ransac(self, plist, iters=100, error=350, good_model_num=10):
        model_error = 255
        model_H = None
        for i in range(iters):
            consensus_set = []
            point_list_tmp = np.copy(plist).tolist()
            # random select 3 points
            for j in range(3):
                temp = choice(point_list_tmp)
                consensus_set.append(temp)
                point_list_tmp.remove(temp)
            fp0, fp1, fp2 = [], [], []
            tp0, tp1, tp2 = [], [], []
            for line in consensus_set:
                fp0.append(line[0][0])
                fp1.append(line[0][1])
                fp2.append(1)

                tp0.append(line[0][0])
                tp1.append(line[0][1])
                tp2.append(1)
            
            fp = np.array([fp0, fp1, fp2], dtype=float)
            tp = np.array([tp0, tp1, tp2], dtype=float)

            H = self.Haffine_from_points(fp, tp)
            
            for p in plist:
                x1, y1 = p[0]
                x2, y2 = p[1]

                A = np.array([x1, y1, 1]).reshape(3, 1)
                B = np.array([x2, y2, 1]).reshape(3, 1)
                out = B - np.dot(H, A)
                dist_err = math.hypot(out[0][0], out[1][0])
                if (dist_err < error):
                    consensus_set.append(p)
            
            if len(consensus_set) >= good_model_num:
                dists = []
                for p in consensus_set:
                    x0, y0 = p[0]
                    x1, y1 = p[1]
                    A = np.array([x0, y0, 1]).reshape(3, 1)
                    B = np.array([x1, y1, 1]).reshape(3, 1)

                    out = B - np.dot(H, A)
                    dist_err = math.hypot(out[0][0], out[1][0])
                    dists.append(dist_err)

                if (max(dists) < error) and ((max(dists)) < model_error):
                    model_error = max(dists)
                    model_H = H
        return model_H

    def Haffine_from_points(self, fp,tp):
        """ find H, affine transformation, such that
            tp is affine transf of fp"""

        if fp.shape != tp.shape:
            raise RuntimeError

        #condition points
        #-from points-
        m = np.mean(fp[:2], axis=1)
        maxstd = max(np.std(fp[:2], axis=1))
        C1 = np.diag([1/maxstd, 1/maxstd, 1])
        C1[0][2] = -m[0]/maxstd
        C1[1][2] = -m[1]/maxstd
        fp_cond = np.dot(C1,fp)

        #-to points-
        m = np.mean(tp[:2], axis=1)
        C2 = C1.copy() #must use same scaling for both point sets
        C2[0][2] = -m[0]/maxstd
        C2[1][2] = -m[1]/maxstd
        tp_cond = np.dot(C2,tp)

        #conditioned points have mean zero, so translation is zero
        A = np.concatenate((fp_cond[:2],tp_cond[:2]), axis=0)
        U,S,V = linalg.svd(A.T)

        #create B and C matrices as Hartley-Zisserman (2:nd ed) p 130.
        tmp = V[:2].T
        B = tmp[:2]
        C = tmp[2:4]

        tmp2 = np.concatenate((dot(C,linalg.pinv(B)),zeros((2,1))), axis=1)
        H = np.vstack((tmp2,[0,0,1]))

        #decondition
        H = np.dot(linalg.inv(C2), np.dot(H,C1))

        return H / H[2][2]

    def affine_transform2(self, im, rot, shift):
        '''
            Perform affine transform for 2/3D images.
        '''
        if np.ndim(im) == 2:
            return ndimage.affine_transform(im, rot, shift)
        else:
            imr = ndimage.affine_transform(im[:, :, 0], rot, shift)
            img = ndimage.affine_transform(im[:, :, 1], rot, shift)
            imb = ndimage.affine_transform(im[:, :, 2], rot, shift)

        return np.dstack((imr, img, imb))
"""
my previous script

        if len(src_pts) > 4:
            self.M, self.mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5)
        else:
            self.M = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
"""