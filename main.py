import cv2
import numpy as np
from matplotlib import pyplot as plt
from interest_point_detect import interest_point_detect
from feature_matching import feature_matching
from panoramic_image_stiching import image_stiching
import sys



fname = sys.argv[1]
interest_point_detect(fname)