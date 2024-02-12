import numpy as np
import math as m
import matplotlib.pyplot as plt
import matplotlib.animation as animation

plt.rcParams['animation.ffmpeg_path'] = 'ffmpeg.exe'


def intercircle(triangle, size1, size, point):
    for i in triangle:
        if ((i[0]-int(size / 2))*(i[0] - int(size / 2))+(i[1] - int(size / 2)) * (i[1] - int(size / 2))) <= ((size - 2*size1) * (size - 2*size1) / 16):
            point[0]=i[0]
            point[1]=i[1]
            return True
    return False
def intersquare(triangle, size1, size, point):
    for i in triangle:
        if i[0]<=size1 or i[0]>=(size-size1) or i[1]<=size1 or i[1]>=(size-size1):
            point[0] = i[0]
            print(i[0])
            print(i[1])
            point[1] = i[1]
            return True
    return False
def pixel(color, img, a):#рисование пикселя
    img[a[0]][a[1]]=color


def bres(x1, y1, x2, y2):


    dx = x2 - x1
    dy = y2 - y1

    is_steep = abs(dy) > abs(dx)


    if is_steep:
        x1, y1 = y1, x1
        x2, y2 = y2, x2

    swapped = False
    if x1 > x2:
        x1, x2 = x2, x1
        y1, y2 = y2, y1
        swapped = True

    dx = x2 - x1
    dy = y2 - y1

    error = int(dx / 2.0)
    ystep = 1 if y1 < y2 else -1


    y = y1
    points = []
    for x in range(x1, x2 + 1):
        coord = (y, x) if is_steep else (x, y)
        points.append(coord)
        error -= abs(dy)
        if error < 0:
            y += ystep
            error += dx

    if swapped:
        points.reverse()
    return points

def shiftMatr(vec):
    mtr = np.array([[1, 0, vec[0]], [0, 1, vec[1]], [0, 0, 1]])
    return mtr

def multipleMatr(mult):
    mtr = np.array([[mult, 0, 0], [0, mult, 0], [0, 0, 1]])
    return mtr
def rotMatr(ang):
    mtr = np.array([[np.cos(ang), -np.sin(ang), 0], [np.sin(ang), np.cos(ang), 0], [0, 0, 1]])
    return mtr

def to_proj_coords(x):
    r,c = x.shape
    x = np.concatenate([x, np.ones((1,c))], axis = 0)
    return x

def to_cart_coords(x):
    x = x[:-1]/x[-1]
    return x
size=256
size1=int(0.1*size)
center=np.array([size/2, size/2])
square=bres(size1,size1, size1, size-size1)+bres(size1,size-size1, size-size1, size-size1)+bres(size-size1,size-size1, size-size1, size1)+bres(size-size1,size1,size1,size1)
rad=np.array([0, (size-2*size1)/4, 1])
n=int(2*np.pi*rad[1])

circle=[]
for i in range(1,n):
    rad1=rad
    rad=rotMatr(2*np.pi/n)@rad
    circle+=bres(int(to_cart_coords(shiftMatr(center)@rad1)[0]), int(to_cart_coords(shiftMatr(center)@rad1)[1]), int(to_cart_coords(shiftMatr(center)@rad)[0]), int(to_cart_coords(shiftMatr(center)@rad)[1]))


c=int(2*rad[1]/(2*m.sqrt(3)))
triangle=np.array([[0, rad[1]/2], [c/2, 0], [-1*c/2, 0]], dtype = np.float32).T

triangle=to_proj_coords(triangle)
triangle=shiftMatr([center[0], center[0]+rad[1]+(rad[1]/4)])@triangle
triangle=to_cart_coords(triangle).astype('int32').T

color1=[255,0,0]
color2=[0,255,0]
color3=[0,0,255]


N1=int(rad[1]*np.pi)
N = 50*int(N1/10)# frames count



frames = []
fig = plt.figure()
center1=(np.sum(triangle.T, axis=1)/3).astype('int32')
tror=triangle
a=np.array([3, -1])
movematr=shiftMatr(a)

mr=np.identity(3)
point=np.array([0,0])
t=1

multiple=1
for k in range(1,N):

    triangle = to_proj_coords(tror.T)
    ang = k*2*np.pi/int(N1/10)
    if k/int(N1/10)==t:
        t+=1

    if t%2==0:
        multiple-=1/int(N1/10)
    else:
        multiple += 1/int(N1/10)





    R = rotMatr(ang)
    mr=mr@movematr
    triangle = mr @ shiftMatr(center1) @ R@ multipleMatr(multiple) @ shiftMatr(-center1) @ triangle
    triangle=to_cart_coords(triangle).T
    triangle=triangle.astype('int32')
    if intercircle(triangle, size1, size, point):
        norm=(point-center)/np.linalg.norm(point-center)
        col=color2
        color2=color3
        color3=col
        a=a-2*(a@norm)*norm
        movematr=shiftMatr(a)
    if intersquare(triangle, size1, size, point):
        norm=point
        if point[0]<=size1:
            norm=np.array([1, 0])/np.linalg.norm(np.array([1, 0]))

        if point[0] >=size-size1:
            norm = np.array([-1, 0])/np.linalg.norm(np.array([-1, 0]))

        if point[1] <=size1:
            norm = np.array([0, 1])/np.linalg.norm(np.array([0, 1]))

        if point[1] >=size-size1:
            norm = np.array([0, -1])/np.linalg.norm(np.array([0, -1]))

        col = color1
        color1 = color3
        color3 = col
        a = a - 2 * (a @ norm) * norm

        movematr = shiftMatr(a)

    img = np.ones((size, size, 3), dtype=np.uint8)*255

    for i in square:
        pixel(color1, img, i)

    for i in circle:
        pixel(color2, img, i)
    tr = bres(triangle[0][0], triangle[0][1], triangle[1][0], triangle[1][1]) + bres(triangle[1][0], triangle[1][1], triangle[2][0], triangle[2][1]) + bres(triangle[2][0], triangle[2][1], triangle[0][0], triangle[0][1])


    for i in tr:
        pixel(color3, img, i)


    im = plt.imshow(img)

    frames.append([im])



ani = animation.ArtistAnimation(fig, frames, interval=40, blit=True, repeat_delay=0)
Writer = animation.writers['ffmpeg']
writer = Writer(fps=24, metadata=dict(artist='Me'), bitrate=1800)
ani.save('tr.mp4', writer)

plt.show()