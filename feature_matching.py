import numpy as np
import cv2
from matplotlib import pyplot as plt

class feature_matching(object):
    def __init__(self, kp1, kp2, des1, des2, img1, img2):
        self.kp1 = kp1
        self.kp2 = kp2
        self.des1 = des1
        self.des2 = des2
        self.img1 = img1
        self.img2 = img2
        self.match_features()
        self.write_img(self.match_img)

    def match_features(self):
        # Match features from keypoints, descriptors, and images
        print("Matching Features...")
        # maybe matcher and match should be implemented by ourself
        matcher = cv2.BFMatcher(cv2.NORM_L2, True)
        self.matches = matcher.match(self.des1, self.des2)
        # self.match_img = self.draw_matches(self.img1, self.kp1, self.img2, self.kp2, self.matches)
        self.match_img = cv2.drawMatches(self.img1, self.kp1, self.img2, self.kp2, self.matches, None)

    def write_img(self, img):
        cv2.imwrite('out-feature_matching.png', img)

    def draw_matches(self, img1, kp1, img2, kp2, matches, inliers = None):
        # Create a new output image that concatenates the two images together
        rows1 = self.img1.shape[0]
        cols1 = self.img1.shape[1]
        rows2 = self.img2.shape[0]
        cols2 = self.img2.shape[1]

        out = np.zeros((max([rows1,rows2]), cols1 + cols2, 3), dtype='uint8')

        # Place the first image to the left
        out[:rows1,:cols1,:] = np.dstack([img1])

        # Place the next image to the right of it
        out[:rows2,cols1:cols1+cols2,:] = np.dstack([img2])

        # For each pair of points we have between both images
        # draw circles, then connect a line between them
        for mat in matches:

            # Get the matching keypoints for each of the images
            img1_idx = mat.queryIdx
            img2_idx = mat.trainIdx

            # x - columns, y - rows
            (x1,y1) = kp1[img1_idx].pt
            (x2,y2) = kp2[img2_idx].pt

            inlier = False

            if inliers is not None:
                for i in inliers:
                    if i.item(0) == x1 and i.item(1) == y1 and i.item(2) == x2 and i.item(3) == y2:
                        inlier = True

            # Draw a small circle at both co-ordinates
            cv2.circle(out, (int(x1),int(y1)), 4, (255, 0, 0), 1)
            cv2.circle(out, (int(x2)+cols1,int(y2)), 4, (255, 0, 0), 1)

            # Draw a line in between the two points, draw inliers if we have them
            if inliers is not None and inlier:
                cv2.line(out, (int(x1),int(y1)), (int(x2)+cols1,int(y2)), (0, 255, 0), 1)
            elif inliers is not None:
                cv2.line(out, (int(x1),int(y1)), (int(x2)+cols1,int(y2)), (0, 0, 255), 1)

            if inliers is None:
                cv2.line(out, (int(x1),int(y1)), (int(x2)+cols1,int(y2)), (255, 0, 0), 1)

        return out

