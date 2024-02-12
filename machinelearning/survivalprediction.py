import pandas as pd
import numpy as np
import time
def sigmoid(x):
    return 1/(1 + np.exp(-x))

def func(x):
    return np.exp(-x)/((1+np.exp(-x))*(1+np.exp(-x)))
def accuracy(X, Y, W, B):
    Z = np.dot(W.T, X) + B
    A = sigmoid(Z)

    A = A > 0.5

    A = np.array(A, dtype='int64')

    acc = (1 - np.sum(np.absolute(A - Y)) / Y.shape[1]) * 100

    print("Accuracy of the model is : ", round(acc, 2), "%")

def constraint(X, Y, W, B):
    n = X.shape[0]
    Z = np.dot(W.T, X) + B
    A = func(Z)
    res = np.dot(A, X.T)
    return res




def model(X, Y, learning_rate, iterations):
    m = X.shape[1]
    n = X.shape[0]

    W = np.zeros((n, 1))
    B = 0



    for i in range(iterations):

        Z = np.dot(W.T, X) + B
        A = sigmoid(Z)




        cost=np.sum((A ** Y) * ((1 - A) ** (1 - Y)))

        dW =  np.dot(Y-A, X.T)
        dB =  np.sum(Y-A)

        W = W + learning_rate * dW.T
        B = B + learning_rate * dB







    return W, B, cost



df=pd.read_excel("БД Титаник.xlsx")

new_header = df.iloc[1]
df = df[2:]
df.columns = new_header

df = df.reset_index(drop=True)

X = df.iloc[:, 1:].values.T
X=X.astype("float32")
Y=df.loc[:,"Выживаемость"].values
Y = Y.reshape(1, X.shape[1])
Y=Y.astype("float32")
print(X.shape)
print(Y.shape)
start_time = time.time()
r=model(X, Y, 0.000023, 30000)
print("%s seconds" % (time.time() - start_time))

print(r[0])
print(r[1])
print(r[2])
print(constraint(X, Y, r[0], r[1]))


