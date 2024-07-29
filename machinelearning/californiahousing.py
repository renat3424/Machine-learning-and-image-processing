import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg as LA
from sklearn.datasets import fetch_california_housing

x, t = fetch_california_housing(return_X_y=True)
N = np.shape(x)[0]
lmbd = 0.001
w = np.random.randn(8) * 0.1


def normalize_x():
    data = np.ones((N, np.shape(x)[1]))
    for i in range(np.shape(x)[1]):
        x_vector = x[:, i]
        max = np.max(x_vector)
        min = np.min(x_vector)
        for j in range(N):
            data[j][i] = (x_vector[j] - min) / (max - min)
    return data


def normalize_t():
    max = np.max(t)
    min = np.min(t)
    for i in range(len(t)):
        t[i] = (t[i] - min) / (max - min)


data = normalize_x()
normalize_t()
set = data
t_set = t
a = 0.0001

train_set, train_t = np.ones((12000, 8)), np.ones(12000)
test_set, test_t = np.ones((N - 12000, 8)), np.ones(N - 12000)

for i in range(12000):
    index = np.random.randint(0, np.shape(set)[0])
    train_set[i] = set[index]
    train_t[i] = t_set[index]
    set = np.delete(set, index, axis=0)
    t_set = np.delete(t_set, index, axis=0)

for i in range(N - 12000):
    index = np.random.randint(0, np.shape(set)[0])
    test_set[i] = set[index]
    set = np.delete(set, index, axis=0)
    test_t[i] = t_set[index]
    t_set = np.delete(t_set, index, axis=0)


def gradient(weight, g_set_x, g_set_t):
    grad = np.zeros(8)
    for j in range(8):
        for i in range(len(g_set_t)):
            grad[j] += (g_set_t[i] - np.dot(weight, g_set_x[i])) * g_set_x[i][j] + lmbd * weight[j]
    return grad * -1


error = np.zeros(1000)


def fit(w_train):
    w_old = w_train
    i = 0
    while i < 1000:
        grad = gradient(w_old, train_set, train_t)
        if LA.norm(grad) < 0.001:
            return w_old
        w_new = w_old - a * grad
        if LA.norm(w_new - w_old) < 0.001:
            return w_new
        w_old = w_new
        for j in range(len(train_t)):
            error[i] += (pow((train_t[j] - np.dot(w_old, train_set[j])), 2))
        error[i] = np.sqrt(error[i] / len(train_t))
        i += 1
        print(i)
    return w_new


w = fit(w)
test_error = 0
t_graph = np.zeros(len(test_t))
for j in range(len(test_t)):
    t_graph[j] = np.dot(w, test_set[j])
    test_error += (test_t[j] - np.dot(w, test_set[j])) ** 2
test_error = np.sqrt(test_error / len(train_t))
error = error[error != 0]
print("Train error: ", error[-1])
print("Test error: ", test_error)
row = np.linspace(1, len(error), len(error))
row2 = np.linspace(1, len(test_t), len(test_t))
# plt.plot(row, error, c="black")

plt.plot(row2, sorted(t_graph), c="red")
plt.plot(row2, sorted(test_t), c="blue")
plt.show()
print()
