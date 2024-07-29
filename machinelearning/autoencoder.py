import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch.optim as optim
import torchvision
import math
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_dataset=torchvision.datasets.MNIST("./", train=True, transform=torchvision.transforms.ToTensor())
test_dataset=torchvision.datasets.MNIST("./", train=False, transform=torchvision.transforms.ToTensor())
dataloader=DataLoader(dataset=train_dataset, batch_size=100, shuffle=True)


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.flatten=nn.Flatten(start_dim=1)
        self.lin=nn.Linear(in_features=28*28, out_features=128)
        self.lin1=nn.Linear(in_features=128, out_features=64)
        self.lin2 = nn.Linear(in_features=64, out_features=49)
        self.relu=nn.ReLU()

    def forward(self, x:torch.Tensor):
        x=self.flatten(x)
        x=self.relu(self.lin(x))
        x = self.relu(self.lin1(x))
        return self.relu(self.lin2(x))

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.lin=nn.Linear(in_features=49, out_features=64)
        self.lin1 = nn.Linear(in_features=64, out_features=28*28)
        self.relu=nn.ReLU()
        self.sigm=nn.Sigmoid()
        self.unflatten=nn.Unflatten(1, (28, 28))

    def forward(self, x:torch.Tensor):
        x=self.relu(self.lin(x))
        x = self.sigm(self.lin1(x))
        return self.unflatten(x)

def train(model, dataloader, Loss, optimizer, num_epochs):
    for epoch in range(num_epochs):
        for i, (x,y) in enumerate(dataloader):
            optimizer.zero_grad()
            x=x.squeeze(dim=1)
            x_pred=model(x)
            loss=Loss(x, x_pred)
            loss.backward()
            optimizer.step()
            print(f"epoch={epoch}, i={i}, loss={loss}")


encoder=Encoder()
decoder=Decoder()
model=nn.Sequential(encoder, decoder)
Loss=nn.MSELoss()
optimizer=optim.Adam(model.parameters())
num_epochs=20
train(model, dataloader, Loss, optimizer, num_epochs)
n=10
dataloader=DataLoader(dataset=test_dataset, batch_size=n, shuffle=True)
with torch.no_grad():
    x, y=next(iter(dataloader))
    x = x.squeeze(dim=1)
    x_pred=model(x)
    print(x.shape, x_pred.shape)
    print(Loss(x_pred, x))
    print(Loss(decoder(encoder(x)), x))
    fig, axs = plt.subplots(2, n)
    for i in range(n):
        axs[0, i].imshow(x[i], cmap="gray")
        axs[0, i].get_xaxis().set_visible(False)
        axs[0, i].get_yaxis().set_visible(False)
        axs[1, i].imshow(x_pred[i], cmap="gray")
        axs[1, i].get_xaxis().set_visible(False)
        axs[1, i].get_yaxis().set_visible(False)

    plt.show()

