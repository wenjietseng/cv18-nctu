import numpy as np
import sys
import skimage as sk
import skimage.io as skio
import skimage.transform as sktr
import time
from scipy.signal import convolve2d as c2d

def split_img(img):
    # convert to double (might want to do this later on to save memory)    
    img = sk.img_as_float(img)
    
    # compute the height of each part (just 1/3 of total)
    height = img.shape[0] // 3

    # separate color channels
    b = img[:height]
    g = img[height: 2 * height]
    r = img[2 * height: 3 * height]
    return [b, g, r]

def crop(image, margin=0.1):
    # remove 10% margin of original image
    height, width = image.shape
    y1, y2 = int(margin * height), int((1 - margin) * height)
    x1, x2 = int(margin * width), int((1 - margin) * width)
    return image[y1:y2, x1:x2]

def ssd(img_1, img_2):
    # image matching metrics - sum-squared difference
    ssd = np.sum((img_1 - img_2) ** 2)
    return ssd

def alignment(img_b, img, rad):
    # a naive 15-pixel implementation, np.roll to move pixels
    mat = np.zeros((rad * 2, rad * 2))
    for i in range(-rad, rad):
        for j in range(-rad, rad):
            img_new = np.roll(img, i, axis=0)
            img_new = np.roll(img_new, j, axis=1)
            ssd_val = ssd(img_b, img_new)
            mat[i + rad, j + rad] = ssd_val
    lowest = mat.argmin()
    row_shift = (lowest // (rad * 2)) - rad
    col_shift = (lowest % (rad * 2)) - rad
    print('row, col shift: ', row_shift, ', ', col_shift)
    return (row_shift, col_shift)

def align_bgr(r, g, b, rad):
    # show the alignment of g and r with respect to b
    bg_align = alignment(b, g, rad)
    br_align = alignment(b, r, rad)
    return (bg_align, br_align)

def auto_contrasting(img, contrast_factor=1.5):
    # auto-contrasting
    median = np.percentile(img, 50)
    img *= contrast_factor
    diff = np.percentile(img, 50) - median
    img -= diff
    img = np.clip(img, -1, 1)
    return img

def main():
    # read in the image
    img = skio.imread(sys.argv[1])
    height = img.shape[0] // 3
    img_split = split_img(img)
    for i in range(len(img_split)):
        img_split[i] = crop(img_split[i])

    # for reconstruct
    original_b = img_split[0]
    original_g = img_split[1]
    original_r = img_split[2]

    # unaligned result
    img_out = np.dstack([img_split[2], img_split[1], img_split[0]])
    # save the image
    fname = './russian-colorizing-output/test-unalign.jpg'
    skio.imsave(fname, img_out)

    # apply sobel operator
    sobel_x = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    sobel_y = sobel_x.T

    for i in range(len(img_split)):
        img_split[i] = c2d(img_split[i], sobel_x)
        img_split[i] = c2d(img_split[i], sobel_y)

    # pyramid speed up
    factor_f = 2
    factor_rad = 2
    f = 20
    rad = int((height // f) // 5)
    total_row_shift_g, total_col_shift_g, total_row_shift_r, total_col_shift_r = 0, 0, 0, 0

    while (f >= 1):
        # down sample n dim image by local averaging
        down_b = sktr.downscale_local_mean(img_split[0], (f, f))
        down_g = sktr.downscale_local_mean(img_split[1], (f, f))
        down_r = sktr.downscale_local_mean(img_split[2], (f, f))
        print('down size: ', down_b.shape)

        # compute similarity
        similar_result = align_bgr(down_r, down_g, down_b, rad)
        bg_result = similar_result[0]
        br_result = similar_result[1]        

        # rolling
        total_row_shift_g += (bg_result[0] * f)
        total_col_shift_g += (bg_result[1] * f)
        total_row_shift_r += (br_result[0] * f)
        total_col_shift_r += (br_result[1] * f)

        img_split[1] = np.roll(img_split[1], bg_result[0] * f, axis=0)
        img_split[1] = np.roll(img_split[1], bg_result[1] * f, axis=1)
        img_split[2] = np.roll(img_split[2], br_result[0] * f, axis=0)
        img_split[2] = np.roll(img_split[2], br_result[1] * f, axis=1)

        # update factors
        f = f // factor_f
        rad = rad // factor_rad
        # print('total_row_shift_g, total_col_shift_g: ', total_row_shift_g, ', ', total_col_shift_g)
        # print('total_row_shift_r, total_col_shift_r: ', total_row_shift_r, ', ', total_col_shift_r)       

    print('total_row_shift_g, total_col_shift_g: ', total_row_shift_g, ', ', total_col_shift_g)
    print('total_row_shift_r, total_col_shift_r: ', total_row_shift_r, ', ', total_col_shift_r)       
    # align_result = align_bgr(img_split[2], img_split[1], img_split[0], rad)
    # bg_result = align_result[0]
    # br_result = align_result[1]
    # print(bg_result, br_result)

    original_g = np.roll(original_g, total_row_shift_g, axis=0)
    original_g = np.roll(original_g, total_col_shift_g, axis=1)
    original_r = np.roll(original_r, total_row_shift_r, axis=0)
    original_r = np.roll(original_r, total_col_shift_r, axis=1)

    img_out = np.dstack([original_r, original_g, original_b])
    img_out = auto_contrasting(img_out)

    # save the alinged image
    fname = './russian-colorizing-output/test-aligned.jpg'
    skio.imsave(fname, img_out)

if __name__ == "__main__":
    start_time = time.time()
    main()
    print("--- %s seconds ---" % (time.time() - start_time))