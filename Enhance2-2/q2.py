import cv2
from matplotlib import pyplot as plt


def change_value(r):
    a = 1.9
    value = (r / 256)
    value **= 0.3
    value *= a
    m = value ** 3
    if abs(value - m) * m > 0.2:
        m = value
    return m * (256 / a)


img = cv2.imread("Enhance2.JPG", 1)
hsv_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

for x in range(img.shape[1]):
    for y in range(img.shape[0]):
        (hsv_img[y, x, 2]) = change_value(hsv_img[y, x, 2])

result = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2RGB)

cv2.imwrite('res02.JPG', result)

imgplot = plt.imshow(cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()
