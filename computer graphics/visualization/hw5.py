import numpy as np
from PIL import Image
import math as m
import matplotlib.pyplot as plt

import matplotlib.animation as animation



plt.rcParams['animation.ffmpeg_path'] = 'ffmpeg.exe'
def normilize(x):
     x= x / np.linalg.norm(x)
     return x
def pixel(color, img, x, y):  # рисование пикселя
    if 0 <= x < img.shape[1] and 0 <= y <img.shape[0]:
        img[x][y] = color


def brezenham(x0, y0, x1, y1, img):  # алгоритм брезенхема

    delta = abs(y1 - y0)
    error = 0.0
    y = y0
    for x in range(x0, x1 + 1):

        print("dot")
        pixel([0, 0, 255], img, x, int(y))
        error = error + delta
        if 2 * error >= abs(x1 - x0):
            y = y + m.copysign(1, (y1 - y0))
            error = error - abs(x1 - x0)


def paint(vertexes, faces, img):  # рисование картинки

    vertexes = vertexes.astype('int32')

    for face in faces:
        x0, y0, z0 = vertexes[face[0, 0], 0], vertexes[face[0, 0], 1], vertexes[face[0, 0], 2]
        x1, y1, z1 = vertexes[face[1, 0], 0], vertexes[face[1, 0], 1], vertexes[face[1, 0], 2]
        x2, y2, z2 = vertexes[face[2, 0], 0], vertexes[face[2, 0], 1], vertexes[face[2, 0], 2]

        brezenham(x0, y0, x1, y1, img)
        brezenham(x1, y1, x2, y2, img)
        brezenham(x1, y1, x2, y2, img)
def  Function(a, b, c):
    return (c[0] - a[0]) * (b[1] - a[1]) - (c[1] - a[1]) * (b[0] - a[0])

def rasterization(vertexes1, vertexes, faces, img, isTexture, cameravector, size, texture):
    z_buffer = np.full((size, size), np.inf)
    for face in faces:
        x0, y0, z0 = vertexes[face[0, 0], 0], vertexes[face[0, 0], 1], vertexes[face[0, 0], 2]
        x1, y1, z1 = vertexes[face[1, 0], 0], vertexes[face[1, 0], 1], vertexes[face[1, 0], 2]
        x2, y2, z2 = vertexes[face[2, 0], 0], vertexes[face[2, 0], 1], vertexes[face[2, 0], 2]
        tx0, ty0, tz0 = vertexes1[face[0, 1], 0], vertexes1[face[0, 1], 1], vertexes1[face[0, 1], 2]
        tx1, ty1, tz1 = vertexes1[face[1, 1], 0], vertexes1[face[1, 1], 1], vertexes1[face[1, 1], 2]
        tx2, ty2, tz2 = vertexes1[face[2, 1], 0], vertexes1[face[2, 1], 1], vertexes1[face[2, 1], 2]
        normal=np.cross([x1 - x0, y1 - y0, z1 - z0], [x2 - x0, y2 - y0, z2 - z0])
        dot=normal@cameravector

        if dot<0:
            continue
        y_min = int(min(y0, y1, y2))
        y_max = int(max(y0, y1, y2))
        x_min = int(min(x0, x1, x2))
        x_max = int(max(x0, x1, x2))
        t = np.linalg.inv(np.array(([x0, x1, x2], [y0, y1, y2], [1, 1, 1])))

        for x_i in range(x_min, x_max + 1):
            for y_i in range(y_min, y_max + 1):
                if not (0 <= x_i < img.shape[1] and 0 <= y_i < img.shape[0]):
                    continue
                x = np.array(([x_i, y_i, 1]))
                v = t @ x
                if not np.all(v >= 0):
                    continue

                z = v@np.array([z0, z1, z2])
                if z < z_buffer[x_i, y_i]:
                    if isTexture==False:
                        col=np.abs(dot*255)
                        pixel([col,col,col], img, x_i, y_i)
                        z_buffer[x_i, y_i]=z
                    else:

                        textcord = [v@np.array([tx0, tx1, tx2]),
                                       v@np.array([ty0, ty1, ty2])]
                        color = texture[int((1 - textcord[1]) * (texture.shape[0] - 1)), int(textcord[0] * (texture.shape[1] - 1))]

                        pixel(color, img, x_i, y_i)
                        z_buffer[x_i, y_i] = z




def shift(c):
    return np.array(([1, 0, 0, c[0]],
                     [0, 1, 0, c[1]],
                     [0, 0, 1, c[2]],
                     [0, 0, 0, 1]))

def scale(a):
    return np.array(([a[0], 0, 0, 0],
                     [0, a[1], 0, 0],
                     [0, 0, a[2], 0],
                     [0, 0, 0, 1]))

def rotateX(alpha):
    return np.array(([1, 0, 0, 0],
                     [0, np.cos(alpha), -np.sin(alpha), 0],
                     [0, np.sin(alpha), np.cos(alpha), 0],
                     [0, 0, 0, 1]))

def rotateY(alpha_rad):
    return np.array(([np.cos(alpha_rad), 0, np.sin(alpha_rad), 0],
                     [0, 1, 0, 0],
                     [-np.sin(alpha_rad), 0, np.cos(alpha_rad), 0],
                     [0, 0, 0, 1]))

def rotateZ(alpha_rad):
    return np.array(([np.cos(alpha_rad), -np.sin(alpha_rad), 0, 0],
                     [np.sin(alpha_rad), np.cos(alpha_rad), 0, 0],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]))

def rotate(alpha_deg):
    alpha_rad = np.radians(alpha_deg)
    return rotateX(alpha_rad[0]) @ rotateY(alpha_rad[1]) @ rotateZ(alpha_rad[2])


def Mo2W(location, alpha_deg, scl):
    T = shift(location)
    R = rotate(alpha_deg)
    S = scale(scl)
    return T @ R @ S

def vec2norm(vec):
    vec_norm = np.linalg.norm(vec)
    if vec_norm == 0:
        vec_norm = 1
    return vec / vec_norm

def rotateC(loc, look):
    gamma = vec2norm((loc - look))
    r=np.array([0, 1, 0])
    beta = vec2norm(r-(r@gamma)/(gamma@gamma)*gamma)
    alpha = vec2norm(np.cross(beta, gamma))
    Rc = np.eye(4)
    Rc[0, :3], Rc[1, :3], Rc[2, :3] = alpha, beta, gamma
    return Rc

def Mw2c(loc, look):
    Tc = shift(-loc)
    Rc = rotateC(loc, look)
    return Rc @ Tc


def Mprojpers(t, b, r, l, f, n):
    return np.array(([2 * n / (r - l), 0, (r + l) / (r - l), 0],
                     [0, 2 * n / (t - b), (t + b) / (t - b), 0],
                     [0, 0, -(f + n) / (f - n), - (2 * f * n / (f - n))],
                     [0, 0, -1, 0]))

def Mprojorth(t, b, r, l, f, n):
    return np.array(([2 / (r - l), 0, 0, -(r + l) / (r - l)],
                     [0, 2 / (t - b), 0, -(t + b) / (t - b)],
                     [0, 0, -2 / (f - n), -(f + n) / (f - n)],
                     [0, 0, 0, 1]))





with open("african_head.obj", 'r') as file:

    obj = [x.split() for x in file.read().splitlines()]
v, vt, vn, f = [], [], [], []  # Grouping
for row in obj:
    if row == []:
        continue
    if row[0] == 'v':
        v.append(row[1:])
    elif row[0] == 'vt':
        vt.append(row[1:])
    elif row[0] == 'vn':
        vn.append(row[1:])
    elif row[0] == 'f':
        f.append([row[1].split('/'), row[2].split('/'), row[3].split('/')])
v = np.array(v, dtype=np.float32)
vt = np.array(vt, dtype=np.float32)
vn = np.array(vn, dtype=np.float32)
fac = np.array(f, dtype=np.int32) - 1
texture = np.asarray(Image.open("african_head_diffuse.tga"), dtype="int32")
v=np.hstack((v, np.ones((v.shape[0], 1))))
vn=np.hstack((vn, np.ones((vn.shape[0], 1))))

size=1024
Mviewport=shift([size/2, size/2, 0])@scale([size / 2, size / 2, 1])

frames = []
fig = plt.figure()
for i in range(50):

    image = np.zeros((size, size, 3), dtype=np.uint8)+255
    pos=[-1, 0, -1]
    rot=[8+360*i/10, 12, 16]
    sca=[0.9, 0.9, 0.9]
    locCam=np.array([2,2,2])
    lookCam=np.array([-2,-2,0])
    cameravector = lookCam - locCam
    v = (Mw2c(locCam, lookCam) @ Mo2W(pos, rot, sca) @ v.T).T
    vn = (np.linalg.inv(Mw2c(locCam, lookCam) @ Mo2W(pos, rot, sca)).T @ vn.T).T

    v = (Mprojorth(2.6, -0.2, 2.6, 0, 32, 12) @ v.T).T
    v= (Mviewport @ v.T).T

    rasterization(vt, v, fac, image, True,cameravector, size, texture)
    print("here")
    img=plt.imshow(image)
    frames.append([img])

ani = animation.ArtistAnimation(fig, frames, interval=40, blit=True, repeat_delay=0)
Writer = animation.writers['ffmpeg']
writer = Writer(fps=24, metadata=dict(artist='Me'), bitrate=1800)
ani.save('tr.mp4', writer)

plt.show()

