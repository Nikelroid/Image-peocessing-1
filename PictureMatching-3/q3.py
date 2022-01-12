import cv2
import numpy as np
from matplotlib import pyplot as plt


def resize(img, zarib):
    w = int(img.shape[1] * zarib)
    h = int(img.shape[0] * zarib)
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)


a2 = 0
b2 = 0
a3 = 0
b3 = 0

img = cv2.imread("master-pnp-prok-01800-01886a.tif", 0)

for count in range(5):

    a2 *= 2
    b2 *= 2
    a3 *= 2
    b3 *= 2

    w = int(img.shape[1] * (0.0625 * (2 ** count)))
    h = int(img.shape[0] * (0.0625 * (2 ** count)))
    dif = int(h / 3)
    gray_img = cv2.resize(img, (w, dif * 3), interpolation=cv2.INTER_AREA)


    if count == 0:
        gray_1 = gray_img[:dif, :w]
        gray_2 = gray_img[dif:dif * 2, :w]
        gray_3 = gray_img[dif * 2:dif * 3, :w]

    q = int(2 ** (4 - (0.4 * count)))
    start_zarib = 0.8
    p = 1.3

    cut_dif_h = int(dif * 3 / (q * 3))
    cut_dif_w = int(w / q)

    a = 1
    b = q - 1

    cut_1 = gray_1[cut_dif_h * a:b * cut_dif_h, cut_dif_w * a:b * cut_dif_w]
    cut_2 = gray_2[cut_dif_h * a:b * cut_dif_h, cut_dif_w * a:b * cut_dif_w]
    cut_3 = gray_3[cut_dif_h * a:b * cut_dif_h, cut_dif_w * a:b * cut_dif_w]

    zarib = 1 - ((1 - start_zarib) / (p ** count))

    z2 = ((1 - zarib) / zarib)
    z1 = z2 / 2

    arz = int(cut_1.shape[1] * zarib)
    tool = int(cut_1.shape[0] * zarib)

    check_area = np.array((cut_1[int(tool * z1):int(tool * z1) + tool, int(arz * z1):int(arz * z1) + arz]),
                          dtype='int16')

    minimum = 100000000000
    best_1 = [0, 0]
    for i2 in range(int(tool * z2)):
        for j2 in range(int(arz * z2)):
            c = cut_2[i2:i2 + tool, j2:j2 + arz]
            if np.sum(np.abs(check_area - c)) < minimum:
                minimum = np.sum(np.abs(check_area - c))
                best_1 = [i2 - int(tool * z1), j2 - int(arz * z1)]

    minimum = 100000000000000
    best_3 = [0, 0]
    for i2 in range(int(tool * z2)):
        for j2 in range(int(arz * z2)):
            c = cut_3[i2:i2 + tool, j2:j2 + arz]
            c2 = cut_3[best_1[0] + int(tool * z1):best_1[0] + int(tool * z1) + tool,
                 best_1[1] + int(arz * z1):best_1[1] + int(arz * z1) + arz]
            if np.sum(np.abs(check_area - c)) < minimum:
                minimum = np.sum(np.abs(check_area - c))
                best_3 = [i2 - int(tool * z1), j2 - int(arz * z1)]

    a2 += best_1[0]
    b2 += best_1[1]
    a3 += best_3[0]
    b3 += best_3[1]


    gray_1 = gray_img[:dif, :w]

    if b2 > 0:
        raw1 = np.column_stack((gray_img, np.zeros((gray_img.shape[0], b2), dtype="uint8")))
        gray_2 = raw1[dif + a2:2 * dif + a2, b2:w + b2]
    elif b2 <= 0:
        raw1 = np.column_stack((np.zeros((gray_img.shape[0], -b2), dtype="uint8"), gray_img))
        gray_2 = raw1[dif + a2:2 * dif + a2, :w]

    if a3 > 0:
        raw3 = np.row_stack((gray_img, np.zeros((a3, gray_img.shape[1]), dtype="uint8")))
    else:
        raw3 = gray_img.copy()

    if b3 > 0:
        raw4 = np.column_stack((raw3, np.zeros((raw3.shape[0], b3), dtype="uint8")))
        gray_3 = raw4[dif * 2 + a3:dif * 3 + a3, b3:w + b3]
    elif b3 < 0:
        raw4 = np.column_stack((np.zeros((raw3.shape[0], -b3), dtype="uint8"), raw3))
        gray_3 = raw4[dif * 2 + a3:dif * 3 + a3, :w]
    else:
        gray_3 = raw3[dif * 2 + a3:dif * 3 + a3, :w]

    proud_image = cv2.merge((gray_1, gray_2, gray_3))
    cv2.imwrite('Amir_'+str(count)+'.jpg',proud_image)
print(b2,a2,b3,a3)
final_image = cv2.merge((gray_1, gray_2, gray_3))

w = int(final_image.shape[1])
h = int(final_image.shape[0])
r, g, b = cv2.split(final_image)

edge_canny = cv2.Canny(final_image, 10.0, 200.0)

end_x = w
start_x = 0
end_y = h
start_y = 0

for i in range(int(15 * w / 16), int(79 * w / 80)):
    total = np.sum(edge_canny[:, i:i + 5] / 255)
    if total > h / 5:
        end_x = i - 1
        break

for i in range(-int(w / 16), -int(w / 80)):
    i *= -1
    total = np.sum(edge_canny[:, i:i + 5] / 255)

    if total > h / 5:
        start_x = i + 1
        break

for i in range(int(15 * h / 16), int(99 * h / 100)):
    total = np.sum(edge_canny[i:i + 10, :] / 255)
    if total > h / 5:
        end_y = i - 5
        break

for i in range(-int(h / 16), -int(h / 100)):
    i *= -1
    total = np.sum(edge_canny[i:i + 10, :] / 255)
    if total > h / 5:
        start_y = i + 5
        break

result = final_image[start_y:end_y, start_x:end_x]

cv2.imwrite('res05-Train.jpg', result)

imgplot = plt.imshow(cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
plt.show()

cv2.waitKey(0)

cv2.destroyAllWindows()
cv2.waitKey()
