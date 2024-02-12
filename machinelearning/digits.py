import random
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score


digits = load_digits()
K = 10
D = len(digits['data'][0])
objects_number = len(digits['data'])
x = digits['data']
classes = digits['target']


def normalization(data):
    return_data = np.empty(shape=data.shape)
    for i in range(len(data)):
        for j in range(len(data[i])):
            cur = data[i, j]
            min = data.T[j].min()
            max = data.T[j].max()
            if max - min == 0:
                return_data[i, j] = 1
                continue
            return_data[i, j] = 2 * ((cur - min) / (max - min)) - 1
    return return_data


def softmax(x):
    return np.exp(x) / sum(np.exp(x))


def sample_divide(N, x, classes):
    sample_return = np.empty(shape=[N, D])
    classes_return = np.empty(shape=[N, ])
    for i in range(N):
        index = random.randint(0, len(x) - 1)
        row = x[index]
        classes_return[i] = classes[index]
        if i == 0:
            sample_return = np.reshape(row, (1, len(x[0])))
        else:
            sample_return = np.concatenate([sample_return, np.reshape(row, (1, len(x[0])))], axis=0)
        classes_return[i] = classes[index]
        x = np.delete(x, index, axis=0)
        classes = np.delete(classes, index, axis=0)
    return sample_return, classes_return, x, classes


def one_hot_encoding(number):
    ohe = np.zeros(K)
    ohe[number] = 1
    return ohe


def array_2_matrix(array, horizontal):
    if horizontal:
        mtrx = np.empty(shape=(1, len(array)))
        for i in range(len(array)):
            mtrx[0][i] = array[i]
    else:
        mtrx = np.empty(shape=(len(array), 1))
        for i in range(len(array)):
            mtrx[i][0] = array[i]
    return mtrx


def y_return(sample, W, b):
    y = np.empty(shape=[0, K])
    for _x in sample:
        x = array_2_matrix(_x, True)
        y_row = softmax(W @ x.T + b).T
        if len(y) == 0:
            y = y_row
        else:
            y = np.concatenate([y, y_row], axis=0)
    return y


def t_return(classes):
    t = np.empty(shape=[0, K])
    for c in classes:
        t_row = array_2_matrix(one_hot_encoding(int(c)), True)
        if len(t) == 0:
            t = t_row
        else:
            t = np.concatenate([t, t_row], axis=0)
    return t


def grad_W(X, classes, W, b):
    Y = y_return(X, W, b)
    T = t_return(classes)
    lmbda = 0.0001
    # return (T - Y).T @ X + lmbda * W
    return (Y - T).T @ X + lmbda * W
    # return (T - Y).T @ X


def grad_b(X, classes, W, b):
    Y = y_return(X, W, b)
    T = t_return(classes)
    ones = array_2_matrix(np.ones(len(T)), False)
    # return Y - T
    return (Y - T).T @ ones
    # return (T - Y).T @ ones


def e_function(X, classes, W, b):
    return_sum = 0
    for i in range(len(X)):
        x = array_2_matrix(X[i], True)
        t = one_hot_encoding(int(classes[i]))
        y = softmax(W @ x.T + b).T
        sum = 0
        for k in range(K):
            sum += np.log(y[0, k]) * t[k]
        return_sum += sum
    return -return_sum


def accuracy_and_confusion_matrix(W, X, b, actual, isAcc):
    predicted = np.array([])
    for i in range(len(X)):
        x = array_2_matrix(X[i], True)
        y = softmax(W @ x.T + b).T
        predicted_class = np.argmax(y)
        predicted = np.append(predicted, predicted_class)
    if isAcc:
        return accuracy_score(actual, predicted)
    else:
        return confusion_matrix(actual, predicted)


# РЅРѕСЂРјР°Р»РёР·Р°С†РёСЏ РІРµРєС‚РѕСЂРѕРІ
x = normalization(x)

# СЂР°Р·РґРµР»РµРЅРёРµ РІС‹Р±РѕСЂРѕРє
x_train, classes_train, x, classes = sample_divide(int(0.8 * objects_number), x, classes)
x_validation, classes_validation, x, classes = sample_divide(int(0.1 * objects_number), x, classes)
x_test, classes_test, x, classes = sample_divide(len(x), x, classes)

# РёРЅРёС†РёР°Р»РёР·Р°С†РёСЏ СЃР»СѓС‡Р°Р№РЅС‹С… РІРµСЃРѕРІ Рё СЃРґРІРёРіР°
mu, sigma = 0, 1 / D  # sigma = 1 / D
Wi = np.random.normal(mu, sigma, size=(K, D))  # (10, 64)
bi = np.random.normal(mu, sigma, size=(K, 1))  # (10, 1)

# confusion matrix Рё accuracy РґРѕ РѕР±СѓС‡РµРЅРёСЏ РЅР° РІР°Р»РёРґР°С†РёРѕРЅРЅРѕР№ РІС‹Р±РѕСЂРєРµ
print(f'Accuracy РЅР° РІР°Р»РёРґР°С†РёРѕРЅРЅРѕР№ РІС‹Р±РѕСЂРєРµ РґРѕ РѕР±СѓС‡РµРЅРёСЏ:',
      accuracy_and_confusion_matrix(Wi, x_validation, bi, classes_validation, True))
cm_validation = accuracy_and_confusion_matrix(Wi, x_validation, bi, classes_validation, False)
cmd_obj = ConfusionMatrixDisplay(cm_validation)
cmd_obj.plot()

# РіСЂР°РґРёРµРЅС‚РЅС‹Р№ СЃРїСѓСЃРє
learning_rate = 0.0001
eps = 0.001
i = 0
e_func_valid_prev, e_func_valid = 0, 0
accuracies_train, accuracies_valid = np.array([]), np.array([])
iterations = np.array([])
while True:
    Wi_1 = Wi
    bi_1 = bi
    Wi = Wi_1 - learning_rate * grad_W(x_train, classes_train, Wi_1, bi_1)
    bi = bi_1 - learning_rate * grad_b(x_train, classes_train, Wi_1, bi_1)
    if i % 50 == 0:
        # РґР»СЏ train РІС‹Р±РѕСЂРєРё
        acc = accuracy_and_confusion_matrix(Wi, x_train, bi, classes_train, True)
        accuracies_train = np.append(accuracies_train, acc)
        e_func = e_function(x_train, classes_train, Wi, bi)
        # РґР»СЏ validation РІС‹Р±РѕСЂРєРё
        acc_valid = accuracy_and_confusion_matrix(Wi, x_validation, bi, classes_validation, True)
        accuracies_valid = np.append(accuracies_valid, acc_valid)
        e_func_valid = e_function(x_validation, classes_validation, Wi, bi)
        print(f'Iteration {i}: Accuracy train = {acc}, E train = {e_func};  Accuracy validation = {acc_valid}, 'f'E validation = {e_func_valid}')
        e_func_valid_prev = e_func_valid
        iterations = np.append(iterations, i)

    if i == 5000 or e_func_valid > e_func_valid_prev or np.linalg.norm(Wi - Wi_1) < eps or np.linalg.norm(grad_W(x_train, classes_train, Wi_1, bi_1)) < eps:
        print('Р”РѕСЃС‚РёРіРЅСѓС‚Р° РЅР°РёР»СѓС‡С€Р°СЏ РјРѕРґРµР»СЊ')
        break
    i += 1

# РІС‹РІРѕРґ РґР°РЅРЅС‹С…
print("Accuracy РЅР° РІР°Р»РёРґР°С†РёРѕРЅРѕР№ РІС‹Р±РѕСЂРєРµ РїРѕСЃР»Рµ РѕР±СѓС‡РµРЅРёСЏ:", accuracies_valid[-1])

# confusion matrix РїРѕСЃР»Рµ РѕР±СѓС‡РµРЅРёСЏ
cm_train = accuracy_and_confusion_matrix(Wi, x_train, bi, classes_train, False)
cmd_obj = ConfusionMatrixDisplay(cm_train)
cmd_obj.plot()

cm_validation = accuracy_and_confusion_matrix(Wi, x_validation, bi, classes_validation, False)
cmd_obj = ConfusionMatrixDisplay(cm_validation)
cmd_obj.plot()

# РіСЂР°С„РёРєРё
fig = plt.figure()
plt.plot(iterations, accuracies_train, c='red')
plt.ylim(0, 1)

fig1 = plt.figure()
plt.plot(iterations, accuracies_valid, c='blue')
plt.ylim(0, 1)

plt.show()