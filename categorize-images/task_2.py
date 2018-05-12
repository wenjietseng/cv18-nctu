""" Task 2: Bag of SIFT representation wtih NN
    - resizes each image to a small, fixed resolution (16*16). (see task_1.py)
    - You can either resize the images to square while ignoring their aspect ratio
      or you can crop the center square portion out of each image. (I ignore this part)
    - The entire image is just a vector of 16*16 = 256 dimensions.
    - use cv2 to find SIFT feature descriptors, which should be Nx128 (N is the number of features detected)
    - Vector Quantization:
        Do K-means clustering to turn descriptors into groups
        historgram the grouping result, code them into another vector
        after this step, apply classifier as we did in task_1.py
    - Nearest neighbor classifier (see task_1.py)
    - can try with KNN + Cross Validation (see task_1.py)
    - Confusion matrix visualization result (see task_1.py)
"""