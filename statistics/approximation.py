import numpy as np
import matplotlib.pyplot as plt

N=1000
x = np.linspace(0, 1, N)
z = 20*np.sin(2*np.pi * 3 * x) + 100*np.exp(x)
error = 10 * np.random.randn(N)
t = z + error

def approximateY(Power, t):
    F = []
    Power=Power+1
    for i in range(Power):
        F.append(np.power(x, i))

    F = np.array(F).T


    #w=np.linalg.pinv(F)@t
    w= np.linalg.solve(F.T@F, t.T@F)
    #Ax=b
    #x=np.linalg.solve(A,b)

    return F@w.T




plt.figure(0)
plt.plot(x, z, '-r', color='black')
plt.plot(x, t, '*', color='blue')
plt.plot(x, approximateY(1, t), '-', color='red')



plt.figure(1)
plt.plot(x, z, '-r', color='black')
plt.plot(x, t, '*', color='blue')
plt.plot(x, approximateY(8, t), '-', color='red')


plt.figure(2)
plt.plot(x, z, '-r', color='black')
plt.plot(x, t, '*', color='blue')
plt.plot(x, approximateY(100, t), '-', color='red')


powers=range(1, 101)
errors=[]
for i in powers:
    y= approximateY(i, t)
    E = (1 / 2) * np.sum(np.power((t - y), 2))
    errors.append(E)

plt.figure(3)

plt.plot(powers, errors, '-', color='blue')


plt.show()

plt.figure(4)
plt.plot(x, z, '-r', color='black')
plt.plot(x, t, '*', color='blue')
plt.plot(x, approximateY(180, t), '-', color='red')
plt.show()