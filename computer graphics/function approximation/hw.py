import numpy as np
import matplotlib.pyplot as plt
import math as m

r = np.zeros((1024, 1024, 3), dtype=np.uint8) + 255


def pixel(color, img, x, y):
    img[x][y]=color




x=np.random.randint(50, 800, 10)
x=np.sort(x)


y=np.random.randint(50, 800, 10)
d = 20*np.random.rand(10)-10
for i in range(0, 10):
    pixel([0, 255, 0], r, x[i], y[i])
    print(x[i], y[i])
    print(r[x[i], y[i]])

xx=[]
yy=[]
for i in range(len(x)-1):
    M=np.array([[1, x[i], x[i]**2, x[i]**3], [1, x[i+1], x[i+1]**2, x[i+1]**3], [0, 1, 2*x[i], 3*x[i]**2], [0, 1, 2*x[i+1], 3*x[i+1]**2]])
    B=np.array([y[i], y[i+1], d[i], d[i+1]])
    A=np.linalg.inv(M)@B
    for t in range(x[i], x[i+1]):
        
            pixel([0, 0, 255], r, t, int(np.array([1, t, t**2, t**3])@A))
            xx.append(t)
            yy.append(int(np.array([1, t, t**2, t**3])@A))


plt.figure(0)
plt.plot(xx, yy, 'o', color='black');
plt.plot(x, y, 'o', color='green');
plt.show()