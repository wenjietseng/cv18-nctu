import cv2
import numpy as np
from matplotlib import pyplot as plt

# create a ShapeError class to raise dimension issue
class ShapeError(Exception):
    def __init__(self, x):
        self.x = x
    
    def __str__(self):
        return self.x

def normalize_2d_pts(pts):
    """ translates and scales the input (homogeneous) points
        such that the output points are centered
        at origin and the mean distance from the origin is sqrt(2).
    Args:
        pts - a 3xN homogeneous points array
    Return:
        new_pts - a normalized 3xN homogeneous points array
        T - a 3x3 transform matrix
    """
    # check point shape is in 3xN
    if pts.shape[0] != 3:
        raise ShapeError('Input point array must be 3xN')

    finiteind = abs(pts[2]) > np.finfo(float).eps
    pts[0, finiteind] = pts[0, finiteind]/pts[2, finiteind]
    pts[1, finiteind] = pts[1, finiteind]/pts[2, finiteind]
    pts[2, finiteind] = 1

    # Centroid of finite points
    c = [np.mean(pts[0, finiteind]), np.mean(pts[1, finiteind])]

    # Shift origin to centrold
    new_pts_0 = pts[0, finiteind] - c[0]
    new_pts_1 = pts[1, finiteind] - c[1]

    mean_dist = np.mean(np.sqrt(new_pts_0**2 + new_pts_1**2))

    scale = np.sqrt(2) / mean_dist

    '''
        T = [scale   0   -scale*c(1)
            0     scale -scale*c(2)
            0       0      1      ];
    '''
    T = np.eye(3)
    T[0][0] = scale
    T[1][1] = scale
    T[0][2] = -scale*c[0]
    T[1][2] = -scale*c[1]

    new_pts = np.dot(T, pts)
    return new_pts, T

def constraint_matrix(x1, x2):
    """ Solve a system of homogeneous linear equation Af=0
    Args:
        x1, x2 - two 3xN points array
    Return:
        A - a matrix Af=0 for SVD
    """
    npts = x1.shape[1]
    # np.c_ concatenation along the 2nd axis, which is column
    A = np.c_[x2[0]*x1[0], x2[0]*x1[1], x2[0],
              x2[1]*x1[0], x2[1]*x1[1], x2[1],
              x1[0],       x1[1],       np.ones((npts,1), dtype=float) ]
    return A

def eight_point_algorithm(x1, x2):
    """ Construct a fundamental matrix with normalized eight-point-algorithm.
    Args:
        x1, x2 - two sets of homogeneous points array (3xN)
    Return:
        F - a fundamental matrix
    """
    x1, T1 = normalize_2d_pts(x1)
    x2, T2 = normalize_2d_pts(x2)

    A = constraint_matrix(x1, x2)

    # 1. Af=0 implies that the fundamental mat F can be extracted from singular vector of V
    # corresponding to the smallest singular value
    U, S, V = np.linalg.svd(A)
    F = V[:, -1].reshape(3,3).T

    # 2. Resolve det(F) = 0 constraint using SVD. Fundamental matrix should be rank 2
    U, D, V = np.linalg.svd(F)
    D[2] = 0
    F = np.dot(U, np.dot(np.diag(D), V.T))

    # 3. denormalize F = T2.T * F_hat * T1
    F = np.dot(np.dot(T2.T, F), T1)
    return F

def random_partition(n,n_data):
    """ return n random rows of data (and also the other len(data)-n rows)
    Args:
        n - number to sample
        n_data - target data
    Return:
        idx1 - sampled n random target data idxs
        idx2 - the rest of target data idxs
    """
    all_idxs = np.arange( n_data )
    np.random.shuffle(all_idxs)
    idxs1 = all_idxs[:n]
    idxs2 = all_idxs[n:]
    return idxs1, idxs2

def RANSAC(x1, x2, n, iters, thres, d, debug=False):
    """ Find fundamental matrix with RANSAC
    Args:
        x1, x2- two homogeneous points array
        n - the minimum number of data values required to fit the model
        iters - the maximum number of iterations allowed in the algorithm
        thres - a threshold value for determining when a data point fits a model
        d - the number of close data values required to assert that a model fits well to data
    Return:
        best_F - the best fitted fundamental matrix
        best_inliers_idx - a list of best fitted inliers' index
        best_err - the best model error
    """
    k = 0
    best_F = None
    best_inliers_idxs = None
    best_err = np.inf
    npts = x1.shape[1]

    while k < iters:
        maybe_idxs, test_idxs = random_partition(n, npts)
        maybe_inliers_x1 = x1[:, maybe_idxs]
        maybe_inliers_x2 = x2[:, maybe_idxs]
        test_points_x1 = x1[:, test_idxs]
        test_points_x2 = x2[:, test_idxs]

        maybe_F = eight_point_algorithm(maybe_inliers_x1, maybe_inliers_x2)
        test_err = abs(np.sum( np.dot(np.dot(test_points_x2.T, maybe_F), test_points_x1), axis=1)) # axis=1 along row
        also_idxs = test_idxs[test_err < thres]
        also_inliers_x1 = x1[:, also_idxs]
        also_inliers_x2 = x2[:, also_idxs]
        
        if debug:
            print('min test_err: ', test_err.min())
            print('max test_err: ', test_err.max())
            print('mean test_err: ', np.mean(test_err))
            print('iteration %d - find %d also inliers' % (k, len(also_idxs)))
        if len(also_idxs) > d:
            better_x1 = np.concatenate((maybe_inliers_x1, also_inliers_x1), axis=1)
            better_x2 = np.concatenate((maybe_inliers_x2, also_inliers_x2), axis=1)
            better_F = eight_point_algorithm(better_x1, better_x2)
            better_errs = np.sum( np.dot(np.dot(better_x2.T, maybe_F), better_x1), axis=1) 
            this_err = np.mean(better_errs)
            if this_err < best_err:
                best_F = better_F
                best_err = this_err
                best_inliers_idxs = np.concatenate((maybe_idxs, also_idxs))
        k+=1
    if best_F is None:
        raise ValueError("did not reach acceptance criteria")
    return best_F, best_inliers_idxs

# read images
img1 = cv2.imread('./homework3/Mesona1.JPG',0)  #queryimage # left image
img2 = cv2.imread('./homework3/Mesona2.JPG',0)  #trainimage # right image

# sift
sift = cv2.xfeatures2d.SIFT_create()

# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

# FLANN parameters
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)

flann = cv2.FlannBasedMatcher(index_params,search_params)
matches = flann.knnMatch(des1,des2,k=2)

good = []
x = []
xp = []

# ratio test as per Lowe's paper
for i, (m, n) in enumerate(matches):
    if m.distance < 0.8 * n.distance:
        good.append(m)
        x.append(kp2[m.trainIdx].pt)
        xp.append(kp1[m.queryIdx].pt)

x = np.asarray(x)
xp = np.asarray(xp)
# turn feature points x and xp into homogeneous coordinates
h_x = np.ones( (x.shape[0], 3), dtype=float)
h_xp = np.ones( (xp.shape[0], 3), dtype=float)
h_x[:, :2] = x
h_xp[:, :2] = xp
h_x = h_x.T
h_xp = h_xp.T

F, inliers = RANSAC(h_x, h_xp, n=8, iters=1000, thres=300, d=200)

print(F, len(inliers))

inliers_x = h_x[:, inliers]
inliers_xp = h_xp[:, inliers]


points_to_draw = []
img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
for i in range(inliers_x.shape[1]):
    points_to_draw.append((int(inliers_x[0, i]), int(inliers_x[1, i])))

for i in range(len(points_to_draw)):
    img1 = cv2.circle(img2, points_to_draw[i], radius=3, color=(255,0,0), thickness=-1)

cv2.imshow('image', img2)
cv2.waitKey(0)
cv2.destroyAllWindows()



"""
# the given intrinsic parameters provided in hw3
K = np.array([[1.4219, 0.0005, 0.5092],
              [0, 1.4219, 0.3802],
              [0, 0, 0.0010]], dtype=float)
K_inv = np.linalg.inv(K)

#
# I guess K is for checking
#
# convert to normalized coords by pre-multiplying all points with the inverse of calibration matrix
# set first camera's coord to world coord
X = np.dot(K_inv, x1)
Xp = np.dot(K_inv, x2)

m = [np.mean(X[0,:]), np.mean(X[1,:])]
sd = [np.std(X[0,:]), np.std(X[1,:])]
print(m, sd)

m = [np.mean(Xp[0,:]), np.mean(Xp[1,:])]
sd = [np.std(Xp[0,:]), np.std(Xp[1,:])]
print(m, sd)
"""

