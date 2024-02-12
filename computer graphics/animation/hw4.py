import numpy as np
import math as m
import matplotlib.pyplot as plt
import matplotlib.animation as animation

plt.rcParams['animation.ffmpeg_path'] = 'ffmpeg.exe'

def bezie(q1, p2, q2, t, img):
    p=q1*(1-t)*(1-t)+2*p2*(1-t)*t+q2*t*t
    p=p.astype('int32')
    pixel([255,0,0], img, p)
def pixel(color, img, a):#рисование пикселя
    img[a[0]][a[1]]=color


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

fig = plt.figure()

size=256
size1=int(0.1*size)
N=24
R=(size-2*size1)/(2*2.5)
center=np.array([size/2, size/2])

Rad=np.array([0, R, 1])



points=[]
control_points=[]
for i in range(0, N):
    points.append(to_cart_coords(shiftMatr([center[0], center[0]])@Rad).astype('int32').T)
    Rad=rotMatr(2 * np.pi / N) @ Rad

print(points, len(points))

for i in range(0, N):
    control_points.append(((points[i]+points[(i+1)%N])/2).astype('int32'))
dir=1
ims=[]
velup=5
veldown=1
path=0
n=1
vel=2
f=0
for k in range(500):
    img = np.ones((size, size, 3), dtype=np.uint8)*255


    for p in range(0, N):
        vec=points[p]-center

        if p%2==0:

            points[p]+=dir*(vel*vec/np.linalg.norm(vec)*velup).astype('int32')

        else:
            points[p]-=dir*(vel*vec/np.linalg.norm(vec)*veldown).astype('int32')

    path+=velup*vel+veldown*vel
    if(path>=3*R):
        path=0
        f += 1

        if f%2==1:
            dir *= -1
        else:
            var=velup
            velup=veldown
            veldown=var


    for i in range(0, N):
        for t in np.linspace(0, 1):
            r=(i+1)%N

            bezie(control_points[i], points[r], control_points[r], t, img)
    im=plt.imshow(img)
    ims.append([im])

ani = animation.ArtistAnimation(fig, ims, interval=40, blit=True, repeat_delay=0)
Writer = animation.writers['ffmpeg']
writer = Writer(fps=24, metadata=dict(artist='Me'), bitrate=1800)
ani.save('tr1.mp4', writer)
plt.show()

