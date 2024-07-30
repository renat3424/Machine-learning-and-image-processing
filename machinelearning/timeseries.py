import numpy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt









class LSTMPredictor(nn.Module):

    def __init__(self, n_hidden=128):
        super(LSTMPredictor, self).__init__()
        self.n_hidden=n_hidden
        self.lstm1=nn.LSTMCell(1, self.n_hidden)
        self.lstm2=nn.LSTMCell(self.n_hidden, self.n_hidden)
        self.lin1=nn.Linear(int(self.n_hidden), 1)


    def forward(self, x:torch.Tensor, future=0):
        output=None
        outputs=[]
        n_samples=x.size(0)

        h_t=torch.zeros(n_samples, self.n_hidden, dtype=torch.float32)
        c_t=torch.zeros(n_samples, self.n_hidden, dtype=torch.float32)
        h_t2=torch.zeros(n_samples, self.n_hidden, dtype=torch.float32)
        c_t2=torch.zeros(n_samples, self.n_hidden, dtype=torch.float32)


        for input_t in x.split(1,dim=1):
            h_t, c_t=self.lstm1(input_t,(h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            output=self.lin1(h_t2)

        outputs.append(output)
        for i in range(future-1):
            h_t, c_t = self.lstm1(output, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            output = self.lin1(h_t2)
            outputs.append(output)

        outputs=torch.cat(outputs, dim=1)
        return outputs

if __name__=="__main__":
    N = 100
    L = 463
    T = 20
    K = 0.001
    x = np.empty((N, L), np.float32)
    x[:] = np.arange(L) + np.random.randint(-4 * T, T * 4, N).reshape(N, 1)
    y = np.sin(x / 1.0 / T) * np.exp(-K * x)
    y = y.astype(numpy.float32)

    # plt.figure()
    # plt.plot(x[0] / 1.0 / T, y[0])
    # plt.show()




    future=100
    train_input = torch.from_numpy(y[3:, :int(L / 2)])
    train_target = torch.from_numpy(y[3:, int(L / 2):int(L / 2)+future])
    test_input=torch.from_numpy(y[:3, :int(L / 2)])
    test_target=torch.from_numpy(y[:3, int(L / 2):int(L / 2)+future])


    model=LSTMPredictor()
    criterion = nn.MSELoss()
    optimizer=optim.LBFGS(model.parameters(), lr=0.05)
    n_steps=101
    for i in range(n_steps):
        print("Step: ", i)
        def closure():
            optimizer.zero_grad()
            y=model(train_input, future)
            loss=criterion(y, train_target)
            print("loss: ", loss.item())
            loss.backward()
            return loss
        optimizer.step(closure)
        with torch.no_grad():

            pred=model(test_input, future)
            loss = criterion(pred, test_target)
            print("test loss: ", loss.item())
            if i%10==0:
                y=pred.detach().numpy()
                plt.figure(figsize=(12, 6))
                plt.title(f"Step: {i+1}")
                plt.xlabel("x")
                plt.ylabel("y")
                plt.xticks(fontsize=20)
                plt.yticks(fontsize=20)

                def draw(y_i, t_input, t_target):
                    n = t_input.shape[0]
                    plt.plot(np.arange(n), t_input, "r", linewidth=2)
                    plt.plot(np.arange(n, n+future), t_target, "r"+":", linewidth=2)
                    plt.plot(np.arange(n, n + future), y_i, "g" + ":", linewidth=2)
                draw(y[0], test_input[0], test_target[0])
                plt.savefig(f"predict{i+1}.pdf")
                plt.close()



















