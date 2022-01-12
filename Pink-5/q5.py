import time
import cv2
import numpy as np
from matplotlib import pyplot as plt

percent = 0
img = cv2.imread("Pink.JPG", 1)
size = 3
t0 = time.time()
crop_value = int(size / 2)
w = int(img.shape[1])
h = int(img.shape[0])

cut_image = img[crop_value:h - crop_value, crop_value:w - crop_value]

r_method1 = np.array(cv2.blur(cut_image, (size, size)), dtype='uint8')
imgplot = plt.imshow(cv2.cvtColor(r_method1, cv2.COLOR_RGB2BGR))
plt.show()
cv2.imwrite('res07.JPG', r_method1)
t1 = time.time()

blur_2 = np.zeros((h, w, 3), np.uint8)
temp_image = img

up_range = int((size + 1) / 2)
low_range = int(-up_range + 1)

for x in range(crop_value, w - crop_value):
    for y in range(crop_value, h - crop_value):
        mid = [0, 0, 0]
        for q in range(low_range, up_range):
            for p in range(low_range, up_range):
                for rgb in range(3):
                    mid[rgb] += (temp_image[y + q, x + p])[rgb]

        blur_2[y, x] = [mid[0] / (size ** 2), mid[1] / (size ** 2), mid[2] / (size ** 2)]

r_method2 = np.array((blur_2[crop_value:h - crop_value, crop_value:w - crop_value]), dtype='uint8')
imgplot = plt.imshow(cv2.cvtColor(r_method2, cv2.COLOR_RGB2BGR))
plt.show()
cv2.imwrite('res08.JPG', r_method2)

t2 = time.time()
q = 0

image = [0 for yy in range(size ** 2)]

rows = [0 for whole_rows in range(size - 1)]
cols = [0 for whole_cols in range(size - 1)]

r_method3 = np.zeros((h - size + 1, w - size + 1, 3), np.float64)

for i in range(size):
    for j in range(size):
        for q in range(size - 1):
            if i - q > 0:
                rows[q] = h - (i - q)
            else:
                rows[q] = -(i - q)
            if j - q > 0:
                cols[q] = w - (j - q)
            else:
                cols[q] = -(j - q)

        row_deleted = np.delete(img, rows, 0)
        col_deleted = np.delete(row_deleted, cols, 1)

        r_method3 += col_deleted

r_method3 = np.array(np.multiply(r_method3, 1 / size ** 2),dtype='uint8')
imgplot = plt.imshow(cv2.cvtColor(r_method3, cv2.COLOR_RGB2BGR))
plt.show()
cv2.imwrite('res09.JPG', r_method3)

t3 = time.time()
print('First method:' + str(t1 - t0))
print('Second method:' + str(t2 - t1))
print('Third method:' + str(t3 - t2))
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.waitKey()
