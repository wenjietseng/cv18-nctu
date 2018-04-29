import cv2
import numpy as np
from matplotlib import pyplot as plt

class ShapeError(Exception):
    """ create a ShapeError class to raise dimension issue
    """
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

def prepare_epilines(F, inliers_pts, h, w):
    """ compute points for drawing epipolar lines
    Args:
        F - fundamental matrix
        inliers_pts - a 3xN points array of inliers
    Return:
        pairs_for_lines - a list contains all points for epilines
    """

    pairs_for_lines = []
    for p in range(inliers_pts.shape[1]):
        [a, b, c] = np.dot(F, inliers_pts[:, p])
        pts_pairs = []
        # ax + by + c = 0
        # we obtain a, b, c and have to compute 2 points for drawing
        # x = 1/a*(-c-by)
        # y = 1/b*(-c-ax)
        y_x0 = int(1/a*(-c))
        y_xh = int(1/a*(-c-b*h))
        x_y0 = int(1/b*(-c))
        x_yw = int(1/b*(-c-a*w))

        if 0 <= y_x0 and y_x0 <= h:
            pts_pairs.append((0, y_x0))
        if 0 <= y_xh and y_xh <= h:
            pts_pairs.append((0, y_xh))
        if 0 <= x_y0 and x_y0 <= w:
            pts_pairs.append((x_y0, 0))
        if 0 <= x_yw and x_yw <= w:
            pts_pairs.append((x_yw, w))
        if (len(pts_pairs) == 2):
            pairs_for_lines.append(pts_pairs)
    return pairs_for_lines

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
    if m.distance < 0.7 * n.distance:
        good.append(m)
        x.append(kp1[m.queryIdx].pt)
        xp.append(kp2[m.trainIdx].pt)

x = np.asarray(x)
xp = np.asarray(xp)
# turn feature points x and xp into homogeneous coordinates
h_x = np.ones( (x.shape[0], 3), dtype=float)
h_xp = np.ones( (xp.shape[0], 3), dtype=float)
h_x[:, :2] = x
h_xp[:, :2] = xp
h_x = h_x.T
h_xp = h_xp.T

F, inliers = RANSAC(h_x, h_xp, n=8, iters=1000, thres=500, d=30)

inliers_x = h_x[:, inliers]
inliers_xp = h_xp[:, inliers]
print(len(inliers))

h, w = img1.shape
# F * xp is the epipolar line associated with x (l = F * xp)
l = prepare_epilines(F, inliers_xp, h, w)
# F.T * x is the epipolar line associated with xp (lp = F.T * x)
lp = prepare_epilines(F.T, inliers_x, h, w)

print(len(l), len(lp))


# draw interest points and their corresponding epipolar lines
img1_vis = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
img2_vis = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)

# kp on img1, epi lines on img2
kp_to_draw = []
for i in range(inliers_x.shape[1]):
    kp_to_draw.append((int(inliers_x[0, i]), int(inliers_x[1, i])))

for i in range(len(kp_to_draw)):
    img1_vis = cv2.circle(img1_vis, kp_to_draw[i], radius=3, color=(255,0,0), thickness=-1)

for i in range(len(l)):
    img2_vis = cv2.line(img2_vis, l[i][0], l[i][1], (255, 0, 0), 1)

vis = np.concatenate((img1_vis, img2_vis), axis=1)

# cv2.imshow('image', vis)
cv2.imwrite('./out_imgs/kp_and_epipolar.png', vis)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

"""
def drawlines(img1,img2,lines,pts1,pts2):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    r,c = img1.shape
    img1 = cv2.cvtColor(img1,cv2.COLOR_GRAY2BGR)
    img2 = cv2.cvtColor(img2,cv2.COLOR_GRAY2BGR)
    for r,pt1,pt2 in zip(lines,pts1,pts2):
        color = tuple(np.random.randint(0,255,3).tolist())
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        img1 = cv2.line(img1, (x0,y0), (x1,y1), color,1)
        img1 = cv2.circle(img1,tuple(pt1),5,color,-1)
        img2 = cv2.circle(img2,tuple(pt2),5,color,-1)
    return img1,img2

"""
#
# step 4: essential matrix
#
# the given intrinsic parameters provided in hw3
K = np.array([[1.4219, 0.0005, 0.5092],
              [0, 1.4219, 0.3802],
              [0, 0, 0.0010]], dtype=float)

# assume camera matrix P1 is P=[I|0]
P1 = np.array([[1, 0, 0, 0],
               [0, 1, 0, 0],
               [0, 0, 1, 0]], dtype=float)

# E = K1.T * F * K2
E = np.dot(np.dot(K.T, F), K)
U, S, V = np.linalg.svd(E)
m = (S[0]+S[1])/2
E = np.dot(np.dot(U, np.diag((m,m,0))), V)
U, S, V = np.linalg.svd(E)
W = np.array([[0, -1, 0],
              [1,  0, 0],
              [0,  0, 1]], dtype=float)
u3 = U[:,2]
P2_1 = np.c_[np.dot(np.dot(U, W),   V),  u3]
P2_2 = np.c_[np.dot(np.dot(U, W),   V), -u3]
P2_3 = np.c_[np.dot(np.dot(U, W.T), V),  u3]
P2_4 = np.c_[np.dot(np.dot(U, W.T), V), -u3]


#
# step 5: find out most appropriate answer
#



#
# step 6: triangulation to get 3D points
#