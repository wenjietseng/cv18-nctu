""" Task 3: Bag of SIFT representation with SVM
    - use cv2 to find SIFT feature descriptors, which should be Nx128 (N is the number of features detected)
    - Vector Quantization:
        Do K-means clustering to turn descriptors into groups
        historgram the grouping result, code them into another vector
        after this step, apply classifier as we did in task_1.py
    - SVM classifier
    - try with Cross Validation
    - Confusion matrix visualization result (see task_1.py)
"""