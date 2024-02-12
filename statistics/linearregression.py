import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
import scipy.stats as stats
import math
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.linear_model import LinearRegression

data=np.loadtxt("r4z1.csv", delimiter=",", dtype=np.str0)[1:].astype("float32").T
nx=5
ny=7
xl=118.05
xr= 126.05
deltax=(xr-xl)/nx
yl=80.05
yr= 86.05
deltay=(yr-yl)/ny
xedges=[]
yedges=[]

while xl<=(xr+0.00001):
    xedges.append(round(xl, 2))
    xl+=deltax
while yl<=(yr+0.00001):
    yedges.append(round(yl, 2))
    yl+=deltay

print(xedges)
print(yedges)
H, xedges, yedges = np.histogram2d(data[0], data[1], bins=(xedges, yedges))

print(H)
r=np.array(H.sum())
print(r)
H=np.append(H,[np.sum(H,axis=0)],axis=0)


col=np.array([np.sum(H,axis=1)])


H=np.concatenate((H,col.T),axis=1)


print(xedges)
print(yedges)
print(H)

alpha = 0.05

T = 0
print(H.shape[1])
for i in range(H.shape[0]-1):
    for j in range(H.shape[1]-1):
        O = H[i][j]
        E = H[i][-1] * H[-1][j] / H[-1][-1]
        T += (O - E) ** 2 / E



p_value = 1 - stats.chi2.cdf(T, (H.shape[0] - 1) * (H.shape[1] - 1))
conclusion = "Признаки независимы"
if p_value <= alpha:
    conclusion = "Признаки зависимы"

print("Cтатистика критерия", T, " значение p", p_value)
print(conclusion)

model = LinearRegression()
data=np.loadtxt("r4z2.csv", delimiter=",", dtype=np.str0)[1:].astype("float32").T
print(data)
x=data[0].reshape((-1, 1))
y=data[1]
model.fit(x, y)
print('intercept:', model.intercept_)

print('slope:', model.coef_)
x0=(81-model.intercept_)/model.coef_[0]
x1 = np.linspace(min(x),max(x),100)
y1 = model.coef_[0]*x1+model.intercept_
plt.plot(x1, y1, '-r', color='orange')
plt.plot(x, y, '.')
plt.plot(x0, 81, '*', color='red')
plt.title('Graph of linear function')
plt.xlabel('x', color='#1C2833')
plt.ylabel('y', color='#1C2833')
plt.legend(loc='upper left')
plt.grid()
plt.show()
x0=(81-model.intercept_)/model.coef_[0]
print(x0)

