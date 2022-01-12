import cv2
from matplotlib import pyplot as plt


def change_hue(pixel, blur_value):
    hue = pixel[0]
    if 175 > hue > 135:
        hue -= 56
        pixel[0] = hue
    else:
        pixel[2] = blur_value
    return pixel


img = cv2.imread("Flowers.JPG", 1)

w = int(img.shape[1])
h = int(img.shape[0])

hsv_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

x = 0
y = 0

for x in range(w):
    for y in range(h):
        mid = 0
        count = 0
        for q in range(-2, 3):
            for p in range(-2, 3):
                try:
                    mid += (img[y + q, x + p])[2]
                    count += 1
                except:
                    count += 0
        (hsv_img[y, x]) = change_hue(hsv_img[y, x], mid / count)

cv2.imwrite('res06.JPG', cv2.cvtColor(hsv_img, cv2.COLOR_HSV2RGB))

imgplot = plt.imshow(cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR))
plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()
