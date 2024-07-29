import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision
from torch.utils.data import DataLoader
import torch.optim as optim
import matplotlib.pyplot as plt

num_classes=10

train_dataset=torchvision.datasets.MNIST("./", train=True, transform=torchvision.transforms.ToTensor(), target_transform=lambda x: F.one_hot(torch.tensor(x), num_classes))
test_dataset=torchvision.datasets.MNIST("./", train=False, transform=torchvision.transforms.ToTensor(), target_transform=lambda x: F.one_hot(torch.tensor(x), num_classes))
dataloader=DataLoader(dataset=train_dataset, batch_size=100, shuffle=True)
class Encoder(nn.Module):
    def __init__(self, input_shape, output_dim, num_classes):
        super(Encoder, self).__init__()
        input_shape=input_shape[0]*input_shape[1]+num_classes
        self.output_dim=output_dim
        self.flat=nn.Flatten(start_dim=1)

        self.lin=nn.Linear(in_features=input_shape, out_features=256)
        self.dropbatch=nn.Sequential(nn.Dropout(p=0.3), nn.BatchNorm1d(num_features=256))
        self.lin1=nn.Linear(in_features=256, out_features=128)
        self.dropbatch1 = nn.Sequential(nn.Dropout(p=0.3), nn.BatchNorm1d(num_features=128))
        self.mean = nn.Linear(in_features=128, out_features=output_dim)
        self.std = nn.Linear(in_features=128, out_features=output_dim)
        self.relu=nn.ReLU()

    def forward(self, x):
        x=self.flat(x)
        x=self.dropbatch(self.relu(self.lin(x)))
        x=self.dropbatch1(self.relu(self.lin1(x)))
        N=torch.randn(x.shape[0], self.output_dim)
        mean=self.mean(x)
        std=self.std(x)

        return mean+torch.exp(std/2)*N, mean, std



class Decoder(nn.Module):
    def __init__(self, input_dim, output_shape, num_classes):
        super(Decoder, self).__init__()
        self.lin=nn.Linear(in_features=input_dim+num_classes, out_features=128)
        self.dropbatch=nn.Sequential(nn.Dropout(p=0.3), nn.BatchNorm1d(num_features=128))
        self.lin1=nn.Linear(in_features=128, out_features=256)
        self.dropbatch1 = nn.Sequential(nn.Dropout(p=0.3), nn.BatchNorm1d(num_features=256))
        self.lin2=nn.Linear(in_features=256, out_features=output_shape[0]*output_shape[1])
        self.relu=nn.ReLU()
        self.sigm = nn.Sigmoid()
        self.unflat=nn.Unflatten(dim=1, unflattened_size=output_shape)

    def forward(self, x):

        x=self.dropbatch(self.relu(self.lin(x)))
        x=self.dropbatch1(self.relu(self.lin1(x)))
        x=self.sigm(self.lin2(x))
        return self.unflat(x)


class Model(nn.Module):
    def __init__(self, encoder, decoder):
        super(Model, self).__init__()
        self.flat = nn.Flatten(start_dim=1)
        self.encoder=encoder
        self.decoder=decoder

    def forward(self, x, y):
        x = self.flat(x)
        x1 = torch.hstack((x, y))
        x, mean, std=self.encoder(x1)
        x1 = torch.hstack((x, y))
        x=self.decoder(x1)
        return x, mean, std

class VAE_loss(nn.Module):
    def __init__(self):
        super(VAE_loss, self).__init__()

    def forward(self, inputs, targets, mean, std):

        inputs = torch.flatten(inputs, start_dim=1)
        targets = torch.flatten(targets, start_dim=1)
        loss=torch.sum(torch.square((inputs-targets)), dim=1)
        print(loss.mean().item())
        kl_loss=-0.5*torch.sum((1-torch.square(mean)-torch.exp(std)+std), dim=1)
        print(kl_loss.mean().item())

        return torch.mean(kl_loss+loss)

def train(model, dataloader, Loss, optimizer, num_epochs):
    for epoch in range(num_epochs):
        for i, (x,y) in enumerate(dataloader):
            x=x.squeeze(dim=1)
            optimizer.zero_grad()
            x_pred, mean, std=model(x, y)
            loss=Loss(x, x_pred, mean, std)
            loss.backward()
            optimizer.step()
            print(f"epoch={epoch}, i={i}, loss={loss}")


def paint_images(first_num, num):
    dataset=get_dataset(first_num, train_dataset)[:num]
    plt.figure()
    k=1
    x, mean, std=encoder(dataset)
    a=train_dataset.targets==first_num
    x=train_dataset.data[a]
    for i in range(num):
        ax = plt.subplot(num, num, k)
        ax.imshow(x[i], cmap="gray")
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        k+=1
    for i in range(num_classes):
        if not i==first_num:
            a = train_dataset.targets == i
            x = torch.hstack((mean, F.one_hot(train_dataset.targets[a][:num], num_classes)))
            x = decoder(x)
            for i in range(num):
                ax = plt.subplot(num, num, k)
                ax.imshow(x[i], cmap="gray")
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)
                k += 1
    plt.show()


def get_dataset(target, dataset):
    a=dataset.targets==target
    return  torch.hstack((torch.flatten(dataset.data[a], start_dim=1), F.one_hot(dataset.targets[a], num_classes))).to(dtype=torch.float32)
input_shape=train_dataset.data.shape[1:]
encoder=Encoder(input_shape, 2, num_classes)
decoder=Decoder(2, input_shape, num_classes)
model=Model(encoder, decoder)
Loss=VAE_loss()
optimizer=optim.Adam(model.parameters())
num_epochs=5
train(model, dataloader, Loss, optimizer, num_epochs)
with torch.no_grad():
    plt.figure()
    num=1
    for i in range(num_classes-1):
        h, mean, std=encoder(get_dataset(i, test_dataset))
        ax = plt.subplot(3, 3, num)
        ax.scatter(h[:, 0], h[:, 1])
        num+=1
    plt.show()
    n = 5
    total = 2 * n + 1
    plt.figure()
    num = 1
    decoder.eval()
    for i in range(-n, n + 1):
        for j in range(-n, n + 1):
            ax = plt.subplot(total, total, num)
            num += 1
            img = decoder(torch.hstack((torch.tensor([[1 * i / n, 1 * j / n]]), F.one_hot(torch.tensor([2]), num_classes))))
            plt.imshow(img.squeeze(dim=0), cmap="gray")
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

    plt.show()
    paint_images(5, 10)
    paint_images(2, 10)