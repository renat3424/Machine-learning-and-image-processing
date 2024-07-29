import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torchsummary import summary
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class NumDataset(Dataset):

    def __init__(self, X, Y):
        self.X=X
        self.Y=Y

    def __getitem__(self, index):
        return torch.tensor(self.X[index], dtype=torch.float32), torch.tensor(self.Y[index], dtype=torch.float32)

    def __len__(self):
        return len(self.Y)

class BidirectionalGru(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_output):
        super(BidirectionalGru, self).__init__()
        self.num_layers=num_layers
        self.hidden_size=hidden_size
        self.bigru=nn.GRU(input_size=input_size, hidden_size=hidden_size, bidirectional=True, batch_first=True, num_layers=num_layers)
        self.lin=nn.Linear(in_features=2*hidden_size, out_features=num_output)

    def forward(self, x):
        N, *_=x.shape
        h0=torch.zeros(2*self.num_layers, N, self.hidden_size)
        x, h0=self.bigru(x, h0)

        return self.lin(x[:,-1])



def plot_graph(model, data, off, length, N):
    XX=torch.zeros(N)
    XX[:off]=torch.tensor(data[:off])
    for i in range(N-off-1):
        x=torch.diag(torch.hstack((XX[i:off+i], torch.tensor(data[off+i+1:i+length])))).unsqueeze(dim=0).to(device=device, dtype=torch.float32)
        y=model(x)
        XX[off+i]=y.item()
    plt.plot(XX, c="red")
    plt.plot(data[:N], c="green")
    plt.show()



def train(model, dataloader, Loss, optimizer, num_epochs):
    for epoch in range(num_epochs):
        for i, (x,y) in enumerate(dataloader):
            x=x.to(device=device)
            y=y.to(device=device)
            optimizer.zero_grad()
            y_pred=model(x).squeeze(dim=1)
            loss=Loss(y, y_pred)
            loss.backward()
            optimizer.step()
            print(f"epoch={epoch}, i={i}, loss={loss}")



if __name__=="__main__":
    N = 10000
    data = [np.sin(x / 20)+0.1*np.random.randn(1)[0] for x in range(N)]
    off = 10
    length = off * 2 + 1
    X = [np.diag(data[i:off + i] + data[off + i + 1:i + length]) for i in range(N - length)]
    Y = data[off:N - off - 1]
    model=BidirectionalGru(length-1, 2, 1, 1).to(device=device)
    dataset=NumDataset(X, Y)
    dataloader=DataLoader(dataset, batch_size=32, shuffle=False)
    Loss=nn.MSELoss()
    optimizer=optim.Adam(model.parameters(), lr=0.01)
    num_epochs=10
    train(model, dataloader, Loss, optimizer, num_epochs)
    plot_graph(model, data, off, length, 200)
