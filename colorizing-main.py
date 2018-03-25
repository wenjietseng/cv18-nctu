import cv2
import numpy as np
import sys

def get_three_imgs(img):
    # split the original image into b, g, r, three color channels
    img_bgr = []
    idx = img.shape[0] // 3
    for channel in range(3):
        img_zero = np.zeros((idx, img.shape[1], 1), np.uint8)
        for row in range(img_zero.shape[0]):
            for col in range(img_zero.shape[1]):
                if channel == 0:
                    img_zero[row][col] = img[(row + (2 * idx))][col]
                if channel == 1:
                    img_zero[row][col] = img[(row + idx)] [col]
                if channel == 2:
                    img_zero[row][col] = img[row][col]
        img_bgr.append(img_zero)
    return img_bgr

def merge_bgr(img):
    # a naive function to merge 3 color filters
    # bad image quality and too slow, cannot handle size
    img_out = np.zeros((len(img) // 3, len(img[0]) - 20, 3), np.uint8)
    for i in range(len(img_out)):
        for j in range(len(img_out[0])):
            img_out[i][j][2] = img[i + (2 * len(img_out))][j]
            img_out[i][j][1] = img[i + (len(img_out))][j]
            img_out[i][j][0] = img[i][j]
    return img_out


def main():
    if len(sys.argv) < 2:
        print("Pleas enter the path to the image")
        sys.exit(0)
    img = cv2.imread(sys.argv[1], 0)
    # cv2.imshow('image', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # img_lst = get_three_imgs(img)
    # for i in img_lst:
    #     cv2.imshow('image', i)
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows() 
    img_out = merge_bgr(img)
    cv2.imshow('img', img_out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()