import numpy as np



def int_r(num):
    num = int(num + (0.5 if num > 0 else -0.5))
    return num



class MLP:
    def __init__(self, lr, epochs):
        self.lr=lr
        self.epochs=epochs
        self.w1=np.random.randn(3,5)
        self.w2 = np.random.randn(6, 1)


    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_der(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))

    def func(self, X, w):

        X = np.concatenate((np.ones((len(X), 1)), np.array(X)), axis=1)



        return np.matmul(X, w)

    def predict(self, X):
        print(X)
        X=np.array([X])

        x1 = self.func(X, self.w1)

        z1 = self.sigmoid(x1)
        x2 = self.func(z1, self.w2)
        z2 = self.sigmoid(x2)
        return int_r(z2[0,0])

    def train(self, X, y):
        n=len(X)

        y=np.array(y).reshape((4,1))

        for i in range(self.epochs):
            # прямое распространение
            x1=self.func(X, self.w1)

            z1=self.sigmoid(x1)
            x2=self.func(z1, self.w2)
            z1 = np.concatenate((np.ones((len(z1), 1)), np.array(z1)), axis=1)
            z2=self.sigmoid(x2)



            #обратное распространение
            d=z2-y
            d2=np.matmul(z1.T, d)
            d1=np.matmul( np.concatenate((np.ones((len(X), 1)), np.array(X)), axis=1).T, d.dot(self.w2[1:,:].T)*self.sigmoid_der(x1))

            self.w1-=self.lr*(1/n)*d1
            self.w2 -= self.lr * (1 / n) * d2









        return

X=[[0,0], [0,1], [1, 0], [1,1]]
y=[0,1,1,0]
t=MLP(lr=0.09, epochs=15000)
t.train(X, y)
print(t.predict([0,1]))

