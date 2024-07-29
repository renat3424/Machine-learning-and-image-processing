import torch
import torch.nn as nn
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch.optim as optim
import math
import pandas as pd
import matplotlib.pyplot as plt


class WineDataset(Dataset):

    def __init__(self):
        xy = np.loadtxt("wine.csv", delimiter=",", dtype=np.float32, skiprows=1)
        self.y=torch.from_numpy(xy[:,[0]])
        self.x=torch.from_numpy(xy[:,1:])
        self.n_samples=xy.shape[0]

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.n_samples


class MnDataset(Dataset):

    def __init__(self, validation=False, percentage=0.8):

        data = torchvision.datasets.MNIST("./", train=True, transform=torchvision.transforms.ToTensor())
        if not validation:
            self.n_samples = int(percentage*data.targets.shape[0])
            self.x = self.norm_data(data.data[:self.n_samples])
            self.y = data.targets[:self.n_samples]
        else:
            self.n_samples = data.targets.shape[0]-int(percentage * data.targets.shape[0])
            self.x = self.norm_data(data.data[-self.n_samples:])
            self.y = data.targets[-self.n_samples:]
        self.pic_shape = data.data[0].shape
        self.feature_number = self.x.shape[1]
        self.output_number=len(data.classes)




    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.n_samples

    def norm_data(self, data):
        return torch.flatten(data, start_dim=1).to(dtype=torch.float32)/255

    def denorm_data(self, data):
        return torch.unflatten(data, dim=1, sizes=self.pic_shape)*255




class FeedForward(nn.Module):

    def __init__(self, n_input, n_output):
        super(FeedForward, self).__init__()
        self.lin=nn.Linear(n_input, 128)
        self.lin1 = nn.Linear(128, n_output)
        self.relu=nn.ReLU()



    def forward(self, x):
        out=self.lin(x)
        out=self.relu(out)
        out=self.lin1(out)
        return out


if __name__=="__main__":
    dataset=MnDataset(percentage=1)
    batch_size=32
    dataloader=DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    # dataset_val=MnDataset(validation=True, percentage=0.90)
    # dataloader_val = DataLoader(dataset_val, batch_size=len(dataset_val), shuffle=True, num_workers=2)
    print(len(dataset))
    data=torchvision.datasets.MNIST("./", train=False, transform=torchvision.transforms.ToTensor())
    test_data=dataset.norm_data(data.data)
    test_targets=data.targets


    num_epochs=50
    model=FeedForward(dataset.feature_number, dataset.output_number)
    Loss=nn.CrossEntropyLoss()
    optimizer=optim.Adam(model.parameters(), lr=0.001)
    losses=[]
    losses_val=[]
    for epoch in range(num_epochs):

        for i, (x, y) in enumerate(dataloader):
            if i+1==int(dataset.n_samples/batch_size):
                with torch.no_grad():
                    y_pred = model(x)
                    loss = Loss(y_pred, y)
                    losses_val.append(loss)
                break
            y_pred = model(x)
            loss=Loss(y_pred, y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()


            if i + 1 == int(dataset.n_samples / batch_size)-1:
                losses.append(loss.item())
                print(f"epoch {epoch + 1}/{num_epochs}, step {i + 1}/{int(dataset.n_samples / batch_size)}, loss={loss.item()}")




            # for (x1, y1) in iter(dataloader_val):
            #     with torch.no_grad():
            #         y_pred = model(x1)
            #         loss = Loss(y_pred, y1)
            #         losses_val.append(loss)


    y_pred=model(test_data).detach()

    accuracy=torch.sum((test_targets==torch.max(y_pred, dim=1)[1]).to(torch.int16))/test_targets.shape[0]
    print(accuracy.item())
    plt.plot(np.arange(len(losses)), losses, color="b")

    plt.plot(np.arange(len(losses)), losses_val, color="g")
    plt.show()



