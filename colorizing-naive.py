import numpy as np
import sys
import skimage as sk
import skimage.io as skio

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

    img_split = split_img(img)
    for i in range(len(img_split)):
        img_split[i] = crop(img_split[i])

    # unaligned result
    img_out = np.dstack([img_split[2], img_split[1], img_split[0]])
    # save the image
    fname = './russian-colorizing-output/test-unalign.jpg'
    skio.imsave(fname, img_out)

    rad = 15
    align_result = align_bgr(img_split[2], img_split[1], img_split[0], rad)
    bg_result = align_result[0]
    br_result = align_result[1]
    print(bg_result, br_result)

    ag = np.roll(img_split[1], bg_result[0], axis=0)
    ag = np.roll(ag, bg_result[1], axis=1)

    ar = np.roll(img_split[2], br_result[0], axis=0)
    ar = np.roll(ar, br_result[1], axis=1)

    img_out = np.dstack([ar, ag, img_split[0]])
    img_out = auto_contrasting(img_out)

    # save the alinged image
    fname = './russian-colorizing-output/test-aligned.jpg'
    skio.imsave(fname, img_out)

if __name__ == "__main__":
    main()