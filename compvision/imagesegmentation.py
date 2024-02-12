import numpy as np
from PIL import Image
from matplotlib import pyplot as plt


def salt_pepper_remove(matrix):
    offset = int(3 / 2)
    finishx = matrix.shape[0] - offset
    finishy = matrix.shape[1] - offset
    startx=offset
    starty=offset
    for i in range(startx, finishx):
        for j in range(starty, finishy):

            if np.array_equal(matrix[i-offset: i+offset+1, j-offset:j+offset+1], np.array([[255, 255, 255], [255, 0, 255], [255, 255, 255]])):
                matrix[i - offset: i + offset + 1, j - offset:j + offset + 1]=np.array([[255,255, 255], [255, 255, 255], [255, 255, 255]])
            if np.array_equal(matrix[i-offset: i+offset+1, j-offset:j+offset+1], np.array([[0, 0, 0], [0, 255, 0], [0, 0, 0]])):
                matrix[i - offset: i + offset + 1, j - offset:j + offset + 1]=np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])


    return matrix
class Point(object):
    def __init__(self, x, y):
        self.x = int(x)
        self.y = int(y)

    def getX(self):
        return self.x

    def getY(self):
        return self.y
def find_threshold(histarr):
    r=np.sum(histarr)
    histarr=histarr/r
    t=25
    S=255
    i=t
    T=i
    while i<256:
        Q1=np.sum(histarr[0:i+1])

        Q2=np.sum(histarr[i+1:256])

        Mu1=np.dot(np.array(range(1, i+2)), histarr[0:i+1])/Q1

        Mu2=np.dot(np.array(range(i+2, 257)), histarr[i+1:256])/Q2

        O1=np.dot(np.square(np.array(range(1, i+2))-Mu1), histarr[0:i+1])/Q1
        O2=np.dot(np.square(np.array(range(i+2, 257)) - Mu2), histarr[i+1:256])/Q2
        Osquare=(Q1*(O1)+Q2*(O2))/r

        if Osquare<S:
            S=Osquare
            T=i

        i=i+t
    return T


def findLocalMinima(arr):

    mn = []
    n=arr.shape[0]

    if (arr[0] < arr[1]):
        mn.append(0)


    for i in range(1, n - 1):


        if (arr[i - 1] > arr[i] < arr[i + 1]):
            mn.append(i)




    if (arr[-1] < arr[-2]):
        mn.append(n - 1)

    return mn



def selectConnects(neighbor_num):
    connects = [Point(-1, -1), Point(0, -1), Point(1, -1), Point(1, 0), Point(1, 1),Point(0, 1), Point(-1, 1), Point(-1, 0)]
    return connects


def regionGrow(img, mask, seed, thresh, neighbor_num=8, label=1):

    height, weight = img.shape

    connects = selectConnects(neighbor_num)

    seedList = []
    seedList.append(seed)

    while (len(seedList) > 0):
        currentPoint = seedList.pop(0)

        mask[currentPoint.x, currentPoint.y] = label
        for i in range(neighbor_num):
            tmpX = currentPoint.x + connects[i].x
            tmpY = currentPoint.y + connects[i].y
            if tmpX < 0 or tmpY < 0 or tmpX >= height or tmpY >= weight or mask[tmpX, tmpY] != 0:
                continue

            if img[tmpX, tmpY] >= thresh and mask[tmpX, tmpY] == 0:
                mask[tmpX, tmpY] = label
                seedList.append(Point(tmpX, tmpY))
    return mask


fig = plt.figure()

im=np.array(Image.open("peacock-butterfly.jpg"))
halftone_image=(im[:,:,0]/3+im[:,:,1]/3+im[:,:,2]/3)
halftone_image=halftone_image.astype(int)

histarr=np.zeros(256)

for x in halftone_image:
    for y in x:
        histarr[y]+=1

a=find_threshold(histarr)
print(a)
salt_pepper_remove(halftone_image)
halftone_image1=halftone_image.copy()
mask=np.zeros(halftone_image1.shape)
regionGrow(halftone_image1, mask, Point(halftone_image1.shape[0]/2, halftone_image1.shape[1]/2), a)

plt.imshow(mask, cmap="gray")


plt.show()

fig = plt.figure()

plt.bar(range(256), histarr)

plt.show()



ar=findLocalMinima(histarr)
print(len(ar))
mask=np.zeros(halftone_image.shape)
r=int(255/len(ar))
s=r
for i in range(len(ar)):
    regionGrow(halftone_image, mask, Point(halftone_image.shape[0]/2, halftone_image.shape[1]/2), ar[i], 8, ar[i])


fig = plt.figure()

plt.imshow(mask)


plt.show()






