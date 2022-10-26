import numpy as np
import cv2


class Point(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def getX(self):
        return self.x

    def getY(self):
        return self.y


def getGrayDiff(img, currentPoint, tmpPoint):
    return abs(int(img[currentPoint.x, currentPoint.y]) - int(img[tmpPoint.x, tmpPoint.y]))


def selectConnects(p):
    if p != 0:
        connects = [Point(-1, -1), Point(0, -1), Point(1, -1), Point(1, 0), Point(1, 1), \
                    Point(0, 1), Point(-1, 1), Point(-1, 0)]
    else:
        connects = [Point(0, -1), Point(1, 0), Point(0, 1), Point(-1, 0)]
    return connects


def regionGrow(img, seeds, thresh, p=1):
    height, weight = img.shape
    seedMark = np.zeros(img.shape)
    seedList = []
    for seed in seeds:
        seedList.append(seed)
    label = 1
    connects = selectConnects(p)
    while (len(seedList) > 0):
        currentPoint = seedList.pop(0)

        seedMark[currentPoint.x, currentPoint.y] = label
        for i in range(8):
            tmpX = currentPoint.x + connects[i].x
            tmpY = currentPoint.y + connects[i].y
            if tmpX < 0 or tmpY < 0 or tmpX >= height or tmpY >= weight:
                continue
            grayDiff = getGrayDiff(img, currentPoint, Point(tmpX, tmpY))
            if grayDiff < thresh and seedMark[tmpX, tmpY] == 0:
                seedMark[tmpX, tmpY] = label
                seedList.append(Point(tmpX, tmpY))
    return seedMark




img = cv2.imread('E:/fashion0814/DPTN/132.jpg')
msk = cv2.imread("E:/fashion0814/mask/132.png")
msk = np.array(msk[:,:,0], dtype=np.float32)
mask_new = np.zeros((256,176))
mask_new[msk == 1]=1
a = np.nonzero(mask_new)
y_nonzero = a[0]
x_nonzero = a[1]
seeds = [Point(y_nonzero[0], x_nonzero[0])]
for i in range(0, 20):
    seeds.append(Point(y_nonzero[i], x_nonzero[i]))
# seeds = [Point(125, 80)]
binaryImg_r = regionGrow(img[:, :, 0], seeds, 10)
binaryImg_g = regionGrow(img[:, :, 1], seeds, 10)
binaryImg_b = regionGrow(img[:, :, 2], seeds, 10)
binaryImg = (binaryImg_r + binaryImg_g + binaryImg_b)/3
cv2.imshow(' ', binaryImg)
cv2.waitKey(0)
