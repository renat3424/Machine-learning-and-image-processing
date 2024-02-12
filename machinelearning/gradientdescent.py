import matplotlib.pyplot as plt
import numpy as np
import random as rand
import sklearn.datasets as dts

def design_matrix(x, M):
    design_mtrx = []

    for i in range(M + 1):
        design_mtrx.append(np.power(x, i))

    design_mtrx = np.array(design_mtrx).T

    return design_mtrx

def normilize(x):
    return (x-x.min())/(x.max()-x.min())
def Xnormilize(X):
    r=[]
    r.append(X[:, 0])
    for i in range(1, np.shape(X)[1]):
        x = X[:, i]
        x=normilize(x)
        r.append(x)
    return np.array(r)
x= dts.fetch_california_housing(return_X_y=True)[0]

y=dts.fetch_california_housing(return_X_y=True)[1]

x_train, x_test = np.empty(shape=[0, 8]), np.empty(shape=[0, 8])
row_num = len(x)


for i in range(int(row_num * 0.9)):
    index = rand.randint(0, len(x) - 1)
    row = np.reshape(x[index], (1, 8))
    if i == 0:
        x_train = row
    else:
        x_train = np.concatenate([x_train, row], axis=0)
    x = np.delete(x, index, axis=0)

for i in range(len(x)):
    index = rand.randint(0, len(x) - 1)
    row = np.reshape(x[index], (1, 8))
    if i == 0:
        x_test = row
    else:
        x_test = np.concatenate([x_test, row], axis=0)
    x = np.delete(x, index, axis=0)


y_train=y[0:len(x_train)]
y_test=y[len(x_train):]



def costFunctionReg(F, t, w, l=0.001):
    y = F @ w.T
    return (1 / 2) * (np.sum(np.power((t - y), 2)) + l * (w @ w.T))


def gradient_descent_reg(X, y, theta, alpha=0.0001, lamda=0.001, num_iters=1000000):


    J_history=[]
    theta1=theta
    for i in range(num_iters):
        h = X @ theta.T
        if np.linalg.norm(((X.T @ (h - y)) + lamda * theta)) < 0.00001:

            return (theta, J_history, i+1)

        theta1=theta
        theta = theta - alpha * ((X.T @ (h - y)) + lamda * theta)
        J_history.append(costFunctionReg(X, y, theta))
        if np.linalg.norm(theta1 - theta) < 0.00001:

            return (theta, J_history, i+1)

    return (theta, J_history, num_iters)


M=8

y_train = normilize(y_train)
x_train = Xnormilize(x_train)
F=Xnormilize(design_matrix(sum(x_train), M)).T


theta= np.random.normal(0, 0.1, M+1)






tupple=gradient_descent_reg(F, y_train, theta)
plt.figure(0)
plt.plot(range(tupple[2]), tupple[1], '-', color='red')

print("train error: ", costFunctionReg(F, y_train, theta))
x_test = Xnormilize(x_test)
F1=Xnormilize(design_matrix(sum(x_test), M)).T
print("weights: ", theta)
print("test error: ", costFunctionReg(F1, y_test, theta))
plt.show()