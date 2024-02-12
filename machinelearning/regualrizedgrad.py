import matplotlib.pyplot as plt
import numpy as np
import random as rand

f_x=lambda x: x
f_sinx=lambda x: np.sin(x)
f_cos_x=lambda x: np.cos(x)
f_exp=lambda x: np.exp(x)
f_sqrt=lambda x: np.sqrt(x)

functions=[f_x, f_sinx, f_cos_x, f_exp, f_sqrt]
func_names=["polinom", "sine", "cosine", "exponent", "square root"]
def PlanMatr(x, f, M):
    F = []
    for i in range(M + 1):
        F.append(np.power(f(x), i))
    return np.array(F).T



def t_func(x):
    z = 20 * np.sin(2 * np.pi * 3 * x) + 100 * np.exp(x)
    error = 10 * np.random.randn(len(x))
    return z + error

def Error(w, t, l, F):
    y = F @ w.T
    return (1 / 2) * (np.sum(np.power((t - y), 2)) + l * (w @ w.T))



def TrainAndValidate(x_train, x_validation, lmbda, M, f):

    # train
    w_arr = []
    F = PlanMatr(x_train, f, M)
    t = t_func(x_train)
    for l in lmbda:
        w = np.linalg.inv(F.T @ F + l * np.identity(F.shape[1])) @ F.T @ t
        w_arr.append(list(w))

    w_arr = np.array(w_arr)


    k = 0
    E_arr = []
    F = PlanMatr(x_validation, f, M)
    t = t_func(x_validation)
    for w in w_arr:
        E_arr.append(Error(w, t, lmbda[k], F))
        k = k + 1
    lambda_min = lmbda[E_arr.index(min(E_arr))]
    w_min = w_arr[E_arr.index(min(E_arr))]
    return (w_min, lambda_min)


# начальные данные
N = 1000  # размерность вектора характеристик
x = np.linspace(0, 1, N)  # вектор характеристик


#lmbda = [0, 0.000001, 0.0001, 0.001, 0.01, 0.1, 0.5, 1, 5, 10, 50, 100, 500, 1000]  # набор коэффициентов регуляризаций

lmbda = np.linspace(0, 0.0000001, N)
M = 100
# выбор множеств для этапов


x_train, x_validation, x_test = np.array([]), np.array([]), np.array([])
for i in range(800):
    index = rand.randint(0, len(x) - 1)
    x_train = np.append(x_train, x[index])
    x = np.delete(x, index)
for i in range(100):
    index = rand.randint(0, len(x) - 1)
    x_validation = np.append(x_validation, x[index])
    x = np.delete(x, index)
for i in range(100):
    index = rand.randint(0, len(x) - 1)
    x_test = np.append(x_test, x[index])
    x = np.delete(x, index)




lminimum=[]
Eminimums=[]
wminimums=[]
for f in functions:
    (w_min, lambda_min)=TrainAndValidate(x_train, x_validation, lmbda, M, f)
    # test
    F = PlanMatr(x_test, f, M)
    t = t_func(x_test)

    y = F @ w_min.T
    lminimum.append(lambda_min)
    Eminimums.append( (1 / 2) * np.sum(np.power((t - y), 2)))
    wminimums.append(w_min)

print("Error: ", min(Eminimums))
index=Eminimums.index(min(Eminimums))
print("lambda: ", lminimum[index])
print("base function:", func_names[index])
x = np.linspace(0, 1, N)
z = 20 * np.sin(2 * np.pi * 3 * x) + 100 * np.exp(x)
error = 10 * np.random.randn(len(x))
t=z + error
y=PlanMatr(x, functions[index], M)@wminimums[index].T

plt.figure(0)

plt.plot(x, t, '.', color='blue')
plt.plot(x, z, '-r', color='black')
plt.plot(x, y, '-', color='red')

plt.show()

