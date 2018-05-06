import cv2
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import csv

class ShapeError(Exception):
    """ create a ShapeError class to raise dimension issue
    """
    def __init__(self, x):
        self.x = x
    
    def __str__(self):
        return self.x

def find_matches(img1, img2):
    """ using SIFT to find interest points and descriptors in two images
    Args:
        img1, img2 - two images
    Return:
        x, xp - interest points in image coordinate
    """
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
            x.append(kp1[m.queryIdx].pt)
            xp.append(kp2[m.trainIdx].pt)
    
    x = np.asarray(x)
    xp = np.asarray(xp)
    return x, xp

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

    # precondition
    # finiteind = abs(pts[2]) > np.finfo(float).eps
    # pts[0, finiteind] = pts[0, finiteind]/pts[2, finiteind]
    # pts[1, finiteind] = pts[1, finiteind]/pts[2, finiteind]
    # pts[2, finiteind] = 1

    pts[0, :] = pts[0, :]/pts[2, :]
    pts[1, :] = pts[1, :]/pts[2, :]
    pts[2, :] = 1

    # Centroid of finite points
    c = [np.mean(pts[0, :]), np.mean(pts[1, :])]

    # Shift origin to centrold
    new_pts_0 = pts[0, :] - c[0]
    new_pts_1 = pts[1, :] - c[1]

    mean_dist = np.mean(np.sqrt(new_pts_0**2 + new_pts_1**2))

    scale = np.sqrt(2) / mean_dist

    '''
    T = [scale      0  -scale*c(1)
             0  scale  -scale*c(2)
             0      0            1]
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

    # A = np.zeros((npts,9))
    # for i in range(npts):
    #     A[i] = [x1[0,i]*x2[0,i], x1[0,i]*x2[1,i], x1[0,i]*x2[2,i],
    #             x1[1,i]*x2[0,i], x1[1,i]*x2[1,i], x1[1,i]*x2[2,i],
    #             x1[2,i]*x2[0,i], x1[2,i]*x2[1,i], x1[2,i]*x2[2,i] ]

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
    F = np.dot(np.dot(U, np.diag(D)), V)

    # 3. denormalize F = T2.T * F(normalized) * T1
    F = np.dot(np.dot(T2.T, F), T1)
    return F/F[2,2]

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

def get_error(x1, x2, F):
    """ compute error of x F xp 
    """
    Fx1 = np.dot(F, x1)
    Fx2 = np.dot(F, x2)
    denom = Fx1[0]**2 + Fx1[1]**2 + Fx2[0]**2 + Fx2[1]**2
    test_err = (np.diag(np.dot(np.dot(x1.T, F), x2)))**2 / denom
    return test_err


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
        maybe_F, _ = cv2.findFundamentalMat(maybe_inliers_x1.T, maybe_inliers_x2.T, cv2.FM_LMEDS)
        # maybe_F = eight_point_algorithm(maybe_inliers_x1, maybe_inliers_x2)
        # print(test_points_x1.shape, maybe_F.shape, test_points_x2.shape)
        # test_err_0 = np.dot(np.dot(test_points_x1.T, maybe_F), test_points_x2) # axis=1 along row
        # test_err = abs(np.sum( np.dot(np.dot(test_points_x2.T, maybe_F), test_points_x1), axis=1)) # axis=1 along row
        # print(test_err_0.shape, test_err.shape)
        
        # Fx1 = np.dot(maybe_F, test_points_x1)
        # Fx2 = np.dot(maybe_F, test_points_x2)
        # denom = Fx1[0]**2 + Fx1[1]**2 + Fx2[0]**2 + Fx2[1]**2
        # test_err = np.diag(test_err_0)**2 / denom
        test_err = get_error(test_points_x1, test_points_x2, maybe_F)

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
            better_F, _ = cv2.findFundamentalMat(better_x1.T, better_x2.T, cv2.FM_LMEDS)
            # better_F = eight_point_algorithm(better_x1, better_x2)
            better_errs = get_error(better_x1, better_x2, better_F)
            # better_errs = np.sum( np.dot(np.dot(better_x2.T, maybe_F), better_x1), axis=1) 
            this_err = np.mean(better_errs)
            if this_err < best_err:
                best_F = better_F
                best_err = this_err
                best_inliers_idxs = np.concatenate((maybe_idxs, also_idxs))
        k+=1
    if best_F is None:
        raise ValueError("did not reach acceptance criteria")
    return best_F, best_inliers_idxs

def draw_epilines(l, lp, x, xp, img1, img2):
    """ compute points for drawing epipolar lines
    Args:
        F - fundamental matrix
        inliers_pts - a 3xN points array of inliers
    Return:
        pairs_for_lines - a list contains all points for epilines
    """
    img1_vis = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    img2_vis = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
    h, w = img1.shape

    for r, pt_x, pt_xp in zip(l, x.T, xp.T):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        x0, y0 = map(int, [0, -r[2]/r[1]])
        x1, y1 = map(int, [w, -(r[2]+r[0]*w)/r[0]])
        img1_vis = cv2.line(img1_vis, (y0,x0), (y1,x1), color, 1)
        img1_vis = cv2.circle(img1_vis, tuple((int(pt_x[0]), int(pt_x[1]))), 3, color, -1)
        img2_vis = cv2.circle(img2_vis, tuple((int(pt_xp[0]), int(pt_xp[1]))), 3, color, -1)

    vis = np.concatenate((img1_vis, img2_vis), axis=1)
    # cv2.imshow('image', vis)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    cv2.imwrite('./out_imgs/left-statue.png', vis)

    img1_vis = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    img2_vis = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)

    for r, pt_x, pt_xp in zip(lp, x.T, xp.T):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        x0, y0 = map(int, [0, -r[2]/r[1]])
        x1, y1 = map(int, [w, -(r[2]+r[0]*w)/r[1]])
        img2_vis = cv2.line(img2_vis, (x0,y0), (x1,y1), color, 1)
        img1_vis = cv2.circle(img1_vis, tuple((int(pt_x[0]), int(pt_x[1]))), 3, color, -1)
        img2_vis = cv2.circle(img2_vis, tuple((int(pt_xp[0]), int(pt_xp[1]))), 3, color, -1)

    vis = np.concatenate((img1_vis, img2_vis), axis=1)
    # cv2.imshow('image', vis)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    cv2.imwrite('./out_imgs/right-statue.png', vis)

def find_second_camera_mat(K1, K2, F):
    """ Given K1, K2, and F, one can obtain essential matrix E
        decompose E to find four possible solutions of the second camera matrix
    Args:
        K1 - intrinsic matrix of camera 1
        K2 - intrinsic matrix of camera 2
        F - fundamental matrix
    Return:
        P2_1, P2_2, P2_3, P2_4 - four possible solution of camera matrix 2
    """

    # assume camera matrix P1 is P=[I|0]
    P1 = np.array([[1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0]], dtype=float)

    # E = K1.T * F * K2
    E = np.dot(np.dot(K1.T, F), K2)
    U, S, V = np.linalg.svd(E)
    m = (S[0]+S[1])/2
    # E = np.dot(np.dot(U, np.diag((m,m,0))), V)
    U, S, V = np.linalg.svd(E)
    W = np.array([[0, -1, 0],
                  [1,  0, 0],
                  [0,  0, 1]], dtype=float)
    u3 = U[:,2]
    P2_1 = np.c_[np.dot(np.dot(U, W),   V.T),  u3]
    P2_2 = np.c_[np.dot(np.dot(U, W),   V.T), -u3]
    P2_3 = np.c_[np.dot(np.dot(U, W.T), V.T),  u3]
    P2_4 = np.c_[np.dot(np.dot(U, W.T), V.T), -u3]
    return P2_1, P2_2, P2_3, P2_4

def triangulation(P2, x, xp):
    """ Bring points on two images back to 3D coordinates
    Args:
        P2 - the second camera matrix 3x4
        x - homogeneous points in the first image 3xN
        xp - homogeneous points in the first image 3xN
    Returns:
        all_X.T - the 3D coords points 4xN
    """
    # precondition check the third element is 1
    for i in range(x.shape[0]-1):
        x[i,:] = x[i,:] / x[2,:]
        xp[i,:] = xp[i,:] / xp[2,:]

    # assume camera matrix P1 is P=[I|0] 
    P1 = np.array([[1, 0, 0, 0],
                   [0, 1, 0, 0],
                   [0, 0, 1, 0]], dtype=float)
    p1 = P1[0, :]
    p2 = P1[1, :]
    p3 = P1[2, :]
    p1p = P2[0, :]
    p2p = P2[1, :]
    p3p = P2[2, :]

    all_X = np.zeros((x.shape[1], 4), dtype=float)
    for i in range(x.shape[1]):
        u = x[0, i]
        v = x[1, i]
        up = xp[0, i]
        vp = xp[1, i]

        A = np.r_[[u * p3.T - p1.T],
                  [v * p3.T - p2.T],
                  [up * p3p.T - p1p.T],
                  [vp * p3p.T - p2p.T]]

        U, S, V = np.linalg.svd(A)
        X = V[:, -1]
        all_X[i,:] = X
    
    # convert 4xN into 3xN
    for i in range(all_X.shape[1]-1):
        all_X[:,i] = all_X[:,i] / all_X[:,3]
    all_X = all_X[:,:3].T
    return all_X

def check_in_front_camera(_3dpoints, P):
    """ Find out how many points in front of camera
        camera center -R.T t
    """
    C = np.dot(P[:,0:3], P[:,3].T)
    count = 0
    for X in range(_3dpoints.shape[1]):
        if np.dot((X - C), P[:,2].T) > 0:
            count+=1
    return count

def plot3d(_3dpoints):
    """ Plot the reconstructed 3D points
    Args:
        _3dpoints - 3xN array
    Reture:
        a 3d scatter plot
    """
    # check dimensions
    # 3d scatter plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(_3dpoints[0,:], _3dpoints[1,:], _3dpoints[2,:], c='b')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()

def write_data(_3dpts, _2dpts, file_index):
    """ write data as csv
    """
    _3dpts_writer = csv.writer(open('./out_3dpts/task2-3dp-'+ str(file_index) + '.csv', 'w'))
    for i in range(_3dpts.shape[1]):
        _3dpts_writer.writerow([_3dpts[0,i], _3dpts[1,i], _3dpts[2,i]])
    
    _2dpts_writer = csv.writer(open('./out_3dpts/task2-2dp-'+ str(file_index) + '.csv', 'w'))
    for i in range(_2dpts.shape[1]):
        _2dpts_writer.writerow([_2dpts[0,i], _2dpts[1,i]])

# read images (can be a function)
img1 = cv2.imread('./homework3/Statue1.bmp',0)  #queryimage # left image
img2 = cv2.imread('./homework3/Statue2.bmp',0)  #trainimage # right image

# step 1: Find out correspondence across images
x, xp = find_matches(img1, img2)
# turn feature points x and xp into homogeneous coordinates (can be a function)
h_x = np.ones( (x.shape[0], 3), dtype=float)
h_xp = np.ones( (xp.shape[0], 3), dtype=float)
h_x[:, :2] = x
h_xp[:, :2] = xp
h_x = h_x.T
h_xp = h_xp.T

# step 2: estimate fundamental matrix with RANSAC
F, inliers = RANSAC(h_x, h_xp, n=8, iters=3000, thres=5000, d=100, debug=True)
inliers_x = h_x[:, inliers]
inliers_xp = h_xp[:, inliers]

# step 3: draw the interest points and the corresponding epipolar lines
# l = F.T * xp
lines_on_img1 = np.dot(F.T, inliers_xp).T
# l' = F * x
lines_on_img2 = np.dot(F, inliers_x).T
draw_epilines(lines_on_img1, lines_on_img2, inliers_x, inliers_xp, img1, img2)

# step 4: get 4 possible solutions of essential matrix from fundamental matrix
# the given intrinsic parameters provided in hw3
K1 = np.array([[5426.566895, 0.678017, 330.096680],
               [0.000000, 5423.133301, 648.950012],
               [0.000000, 0.000000, 1.000000]], dtype=float)
K2 = np.array([[5426.566895, 0.678017, 387.430023],
               [0.000000, 5423.133301, 620.616699],
               [0.000000, 0.000000, 1.000000]], dtype=float)

P2_1, P2_2, P2_3, P2_4 = find_second_camera_mat(K1, K2, F)

# step 5: find out most appropriate answer
all_3dpoints_1 = triangulation(P2_1, inliers_x, inliers_xp)
all_3dpoints_2 = triangulation(P2_2, inliers_x, inliers_xp)
all_3dpoints_3 = triangulation(P2_3, inliers_x, inliers_xp)
all_3dpoints_4 = triangulation(P2_4, inliers_x, inliers_xp)

# step 6: triangulation to get 3D points of each P2
counts = []
counts.append(check_in_front_camera(all_3dpoints_1, P2_1))
counts.append(check_in_front_camera(all_3dpoints_2, P2_2))
counts.append(check_in_front_camera(all_3dpoints_3, P2_3))
counts.append(check_in_front_camera(all_3dpoints_4, P2_4))
final = np.argmax(counts)

if final == 1:
    print("The first P was selected")
    plot3d(all_3dpoints_1)
    write_data(all_3dpoints_1, inliers_xp, '-statue')
elif final == 2:
    print("The second P was selected")
    plot3d(all_3dpoints_2)
    write_data(all_3dpoints_2, inliers_xp, '-statue')
elif final == 3:
    print("The third P was selected")
    plot3d(all_3dpoints_3)
    write_data(all_3dpoints_3, inliers_xp, '-statue')
else:
    print("The fourth P was selected")
    plot3d(all_3dpoints_4)
    write_data(all_3dpoints_3, inliers_xp, '-statue')