import numpy as np
import matplotlib.pyplot as plt

plt.style.use(["science", "notebook", "grid"])
n=100
matr=np.zeros((4,n))
sigma=3
tau=3

matr[0]=np.random.normal(165, 25, n)
matr[1]=np.random.normal(62, 100, n)
print(matr[0], matr[1])
i=0

while i<n:
    if 85<=matr[0][i]<=210 and 30<=matr[1][i]<=170:
        i=i+1
    else:
        matr[0][i]=np.random.normal(165, 25)
        matr[1][i] = np.random.normal(62, 100)

sigmas=np.linspace(sigma, 60, 1000)
taus=np.linspace(tau, 10, 100)
illsigma=np.zeros(1000)
illtau=np.zeros(100)
print(matr[0])
print(matr[1])
matr[2]=matr[1]/matr[0]
print(matr[1]/matr[0])

for i in range(0, sigmas.size):
    matr[2]=matr[1]/matr[0]+np.random.normal(0, sigmas[i], n)
    matr[3]=np.where(matr[2]<tau, 0, 1)
    illsigma[i]=matr[3][[matr[3]==1]].size


for i in range(0, taus.size):
    matr[2]=matr[1]/matr[0]+np.random.normal(0, sigma, n)
    matr[3]=np.where(matr[2]<taus[i], 0, 1)
    illtau[i]=matr[3][[matr[3]==1]].size

plt.figure(1, figsize=(10,3.5))
plt.plot(range(1, n+1), matr[0], '-')
plt.ylabel("Рост[См]")
plt.xlabel("Люди")
plt.figure(2, figsize=(10,3.5))
plt.plot(range(1, n+1), matr[1], '-')
plt.ylabel("Вес[Кг]")
plt.xlabel("Люди")
plt.figure(3, figsize=(10,3.5))
hist=plt.hist(matr[1]/matr[0], bins=int(n/10),   edgecolor="black")
plt.figure(4, figsize=(10,3.5))
plt.scatter(taus, illtau)
plt.figure(5, figsize=(10,3.5))
plt.scatter(sigmas, illsigma)
plt.show()

