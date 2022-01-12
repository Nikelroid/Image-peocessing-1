import time

import cv2
import numpy as np
from matplotlib import pyplot as plt




def plot_histogram(img):
    color = ('b', 'g', 'r')
    for i, col in enumerate(color):
        histr = cv2.calcHist([img], [i], None, [256], [0, 256])
        plt.plot(histr, color=col)
        plt.xlim([0, 256])
    plt.show()


start = time.time()

dark = cv2.imread("Dark.jpg", 1)
w_dark = int(dark.shape[1])
h_dark = int(dark.shape[0])
org_pink = cv2.imread("Pink.jpg", 1)
pink = cv2.resize(org_pink, (w_dark, h_dark), interpolation=cv2.INTER_AREA)

for rgb in range(3):
    difference = [0] * 256
    Mapper = [[] for x in range(256)]

    for i in range(w_dark):
        for j in range(h_dark):
            difference[dark[j, i, rgb]] += 1
            difference[pink[j, i, rgb]] -= 1
            Mapper[dark[j, i, rgb]].append(int((j * w_dark) + i))



    for i in range(255):
        if difference[i] < len(Mapper[i]):
            Mapper[i + 1].extend(mat[:difference[i]])
            dark[np.array(np.floor(np.divide(mat, w_dark)), 'i'), np.array(np.mod(mat, w_dark), 'i'), rgb] = i + 1
            Mapper[i] = []
            difference[i + 1] += difference[i]
            difference[i] = 0
        else:
            Mapper[i + 1].extend(mat)
            dark[np.array(np.floor(np.divide(mat, w_dark)), 'i'), np.array(np.mod(mat, w_dark), 'i'), rgb] = i + 1
            difference[i + 1] += len(mat)
            Mapper[i] = []
            difference[i] = 0
        print(str(i) + ' - ' + str(rgb))

plot_histogram(dark)
plot_histogram(org_pink)
cv2.imwrite('res10.jpg', dark)
end = time.time()

print(end - start)

cv2.waitKey(0)
cv2.destroyAllWindows()
