from sklearn.datasets import load_digits
from matplotlib import pyplot as plt
import numpy as np
from sklearn.utils import shuffle


def one_hot(y, c):

    y_hot = np.zeros((len(y), c))


    y_hot[np.arange(len(y)), y] = 1

    return y_hot


def softmax(z):
    # z--> linear part.

    # subtracting the max of z for numerical stability.
    exp = np.exp(z - np.max(z))

    # Calculating softmax for all examples.
    for i in range(len(z)):
        exp[i] /= np.sum(exp[i])

    return exp


def accuracy(y, y_hat):
    return np.sum(y==y_hat)/len(y)

def predict(X, w, b):
    # X --> Input.
    # w --> weights.
    # b --> bias.

    # Predicting
    z = X @ w + b
    y_hat = softmax(z)

    # Returning the class with highest probability.
    return np.argmax(y_hat, axis=1)
def fit(X, y, lr, c, epochs):
    # X --> Input.
    # y --> true/target value.
    # lr --> Learning rate.
    # c --> Number of classes.
    # epochs --> Number of iterations.

    # m-> number of training examples
    # n-> number of features
    m, n = X.shape

    # Initializing weights and bias randomly.
    w = np.random.random((n, c))
    b = np.random.random(c)
    # Empty list to store losses.
    losses = []

    # Training loop.
    for epoch in range(epochs):

        # Calculating hypothesis/prediction.
        z = X @ w + b
        y_hat = softmax(z)

        # One-hot encoding y.
        y_hot = one_hot(y, c)

        # Calculating the gradient of loss w.r.t w and b.
        w_grad = (1 / m) * np.dot(X.T, (y_hat - y_hot))
        b_grad = (1 / m) * np.sum(y_hat - y_hot)

        # Updating the parameters.
        w = w - lr * w_grad
        b = b - lr * b_grad

        # Calculating loss and appending it in the list.
        loss = -np.mean(np.log(y_hat[np.arange(len(y)), y]))
        losses.append(loss)
        # Printing out the loss at every 100th iteration.
        acuracy=accuracy(predict(X, w, b), y)
        if epoch % 100 == 0:
            print('Epoch {epoch}==> Loss = {loss}'
                  .format(epoch=acuracy, loss=loss))

    return w, b, losses


def standartize(x):
    x_med = x.sum(axis=0) / len(x)
    sigma = np.sqrt(np.sum(np.square(x - x_med) / len(x_med), axis=1))

    i=0
    for x_i in x:
        if sigma[i]!=0:
            x[i]=((x_i-x_med)/sigma[i])
        i=i+1

    return x

digits=load_digits()

x=standartize(digits.data)

y=digits.target

x,y=shuffle(x, y)

number_of_test=int(0.8*len(x))

x_train=x[:number_of_test]
y_train=y[:number_of_test]

x_test=x[number_of_test:]
y_test=y[number_of_test:]


w, b, l = fit(x_train, y_train, lr=0.9, c=10, epochs=1000)







