import numpy as np
import matplotlib.pyplot as plt
import math as m

def pixel(color, img, x, y):#рисование пикселя
    img[x][y]=color

def brezenham(x0,y0,x1,y1, img):#алгоритм брезенхема


    delta = abs(y1-y0)
    error = 0.0
    y = y0
    for x in range(x0, x1 + 1):
        d=(1-m.sqrt(np.dot([x, y], [len(img)/2, len(img[0])/2]))/len(img))
        
        pixel([0, 0, int(255*d)], img, x, int(y))
        error = error + delta
        if 2*error >= abs(x1-x0):

            y = y + m.copysign(1, (y1-y0))
            error = error - abs(x1-x0)


def paint(vertexes, faces, img):#рисование картинки
    coordinates=np.array(vertexes)[:, [0,1]]*145

    coordinates[:,0]+=len(img)/2
    coordinates[:,1] += len(img[0]) / 2
    coordinates=coordinates.astype('int32')

    for face in faces:

        brezenham(coordinates[face[0]-1][0], coordinates[face[0]-1][1], coordinates[face[1]-1][0], coordinates[face[1]-1][1], img)
        brezenham(coordinates[face[1]-1][0], coordinates[face[1]-1][1], coordinates[face[2]-1][0], coordinates[face[2]-1][1], img)
        brezenham(coordinates[face[0]-1][0], coordinates[face[0]-1][1], coordinates[face[2]-1][0], coordinates[face[2]-1][1], img)


with open("teapot.obj", "r") as f:
    data=f.readlines()

    vertexes=[]
    faces=[]
    for i in range(0, len(data)):
        el=data[i]
        if el[0]=="v":
            vertexes.append(list(map(float, el.replace("\n", "").split(" ")[1:])))
        elif el[0]=="f":
            faces.append(list(map(int, el.replace("\n", "").split(" ")[1:])))
    r = np.zeros((1024, 1024, 3), dtype=np.uint8)+255

    paint(vertexes, faces, r)
    r=np.rot90(r)

    plt.figure(0)
    plt.imshow(r)
    plt.imsave("teapot.png", r)
    plt.show()