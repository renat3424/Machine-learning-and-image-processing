import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

def Gauss(x, y, sigma):
    return np.exp(-1*(x**2+y**2)/(2*sigma**2))/(2*np.pi*sigma**2)


def Filter(dim, sigma):
    filter=np.zeros((dim,dim))
    for i in range(dim):
        for j in range(dim):
            filter[i,j]=Gauss((dim-1)/2-i, j-(dim-1)/2, sigma)

    return filter/filter.sum()

def GaussCore(dim, array, sigma):
    array1=np.zeros((array.shape[0]+dim-1, array.shape[1]+dim-1, array.shape[2]))
    for i in range(int((dim-1)/2), int(array.shape[0]+(dim-1)/2)):
        for j in range(int((dim - 1) / 2), int(array.shape[1] + (dim - 1) / 2)):
            array1[i, j, 0]=array[i-int((dim - 1) / 2), j-int((dim - 1) / 2) , 0]
            array1[i, j, 1] = array[i-int((dim - 1) / 2), j-int((dim - 1) / 2) , 1]
            array1[i, j, 2] = array[i-int((dim - 1) / 2), j-int((dim - 1) / 2) , 2]
    for i in range(int((dim-1)/2), int(array.shape[0]+(dim-1)/2)):
        for j in range(int((dim - 1) / 2), int(array.shape[1] + (dim - 1) / 2)):
            i1=i-int((dim-1)/2)
            j1=j-int((dim-1)/2)
            filter=Filter(dim, sigma)
            S1=0
            S2=0
            S3=0

            for i2 in range(dim):
                for j2 in range(dim):
                    S1+=array1[i2+i1, j2+j1, 0]*filter[i2, j2]
                    S2+=array1[i2+i1, j2+j1, 1]*filter[i2, j2]
                    S3+=array1[i2+i1, j2+j1, 2]*filter[i2, j2]

            array1[i, j, 0]=int(S1)

            array1[i, j, 1]=int(S2)
            array1[i, j, 2] = int(S3)

    array2=np.zeros(array.shape)
    for i in range(int((dim-1)/2), int(array.shape[0]+(dim-1)/2)):
        for j in range(int((dim - 1) / 2), int(array.shape[1] + (dim - 1) / 2)):
            array2[i-int((dim - 1) / 2), j-int((dim - 1) / 2) , 0]=array1[i, j, 0]
            array2[i-int((dim - 1) / 2), j-int((dim - 1) / 2) , 1]=array1[i, j, 1]
            array2[i-int((dim - 1) / 2), j-int((dim - 1) / 2) , 2]=array1[i, j, 2]
    return array2





fig = plt.figure(figsize=(10, 7))
fig.add_subplot(3, 3, 1)
im=np.array(Image.open("peacock-butterfly.jpg"))
plt.imshow(im)



im1=GaussCore(5, im, 1.5)
im1=im1.astype(int)
fig.add_subplot(3, 3, 2)
plt.imshow(im1)



fig.add_subplot(3, 3, 3)
img2=im-im1
img2=img2.astype(int)
plt.imshow(img2)


invert_image=np.ones(im.shape)*255-im
invert_image=invert_image.astype(int)

fig.add_subplot(3, 3, 4)
plt.imshow(invert_image)

halftone_image=(im[:,:,0]/3+im[:,:,1]/3+im[:,:,2]/3)

halftone_image=halftone_image.astype(int)

fig.add_subplot(3, 3, 5)
plt.imshow(halftone_image, cmap="gray")


noisyimg=(im+np.random.normal(10, 2, im.shape))%255
noisyimg=noisyimg.astype(int)

fig.add_subplot(3, 3, 6)
plt.imshow(noisyimg)


histarr=np.zeros(256)

for x in halftone_image:
    for y in x:
        histarr[y]+=1
fig.add_subplot(3, 3, 7)
plt.bar(range(256), histarr)


cumsum=np.cumsum(histarr)
cumsum=((cumsum-cumsum.min())*255)/(cumsum.max()-cumsum.min())
cumsum=cumsum.astype(int)
newimg=cumsum[halftone_image.flatten()]
newimg=np.reshape(newimg, halftone_image.shape)
fig.add_subplot(3, 3, 8)
plt.imshow(newimg, cmap="gray")
fig.add_subplot(3, 3, 9)
histarr=np.zeros(256)

for x in newimg:
    for y in x:
        histarr[y]+=1
plt.bar(range(256), histarr)
plt.show()






