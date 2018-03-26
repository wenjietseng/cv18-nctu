import numpy as np
import sys
import skimage as sk
import skimage.io as skio

# read in the image
img = skio.imread(sys.argv[1])

# convert to double (might want to do this later on to save memory)    
# img = img.astype(float)
img = sk.img_as_float(img)
    
# compute the height of each part (just 1/3 of total)
height = img.shape[0] // 3

# separate color channels
b = img[:height]
g = img[height: 2 * height]
r = img[2 * height: 3 * height]

def crop(image, margin=0.1):
    height, width = image.shape
    y1, y2 = int(margin * height), int((1 - margin) * height)
    x1, x2 = int(margin * width), int((1 - margin) * width)
    return image[y1:y2, x1:x2]

b = crop(b)
g = crop(g)
r = crop(r)

# unaligned result
img_out = np.dstack([r, g, b])
# save the image
fname = './russian-colorizing-output/unalign-01522v.jpg'
skio.imsave(fname, img_out)

# align the images
# functions that might be useful for aligning the images include:
# np.roll, np.sum, sk.transform.rescale (for multiscale)

#naive 15-pixel implementation
def ssd(img_1, img_2):
    ssd = np.sum((img_1 - img_2) ** 2)
    return ssd

def alignment(img_b, img, rad):
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
    bg_align = alignment(b, g, rad)
    br_align = alignment(b, r, rad)
    return (bg_align, br_align)

rad = 15
align_result = align_bgr(r, g, b, rad)
bg_result = align_result[0]
br_result = align_result[1]
print(bg_result, br_result)

ag = np.roll(g, bg_result[0], axis=0)
ag = np.roll(ag, bg_result[1], axis=1)

ar = np.roll(r, br_result[0], axis=0)
ar = np.roll(ar, br_result[1], axis=1)

img_out = np.dstack([ar, ag, b])

# auto-contrasting
median = np.percentile(img_out, 50)
contrast_factor = 1.5
img_out *= contrast_factor
diff = np.percentile(img_out, 50) - median
img_out -= diff
img_out = np.clip(img_out, -1, 1)

# save the image
fname = './russian-colorizing-output/aligned-01522v.jpg'
skio.imsave(fname, img_out)
# skio.imshow(img_out)
# skio.show()