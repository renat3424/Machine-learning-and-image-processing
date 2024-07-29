import numpy
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import Dataset, DataLoader
import random
import matplotlib.pyplot as plt
import time
import torch.optim as optim
train_dataset=torchvision.datasets.MNIST("./", train=True, transform=torchvision.transforms.ToTensor(), target_transform=lambda x: F.one_hot(torch.tensor(x), num_classes))



def get_dataset(target, dataset, batch_size, shuffle=True):
    a=dataset.targets==target
    data=dataset.data[a]
    if shuffle==True:
        c = [i for i in range(data.shape[0])]
        random.shuffle(c)
        data=data[c]
    data=data.unsqueeze(dim=1)
    for i in range(0, data.shape[0], batch_size):
        yield data[i:i+batch_size]/255

def get_dataset_len(target, dataset):
    a = dataset.targets == target
    return dataset.data[a].shape[0]
class Generator(nn.Module):
    def __init__(self, hidden_dim):
        super(Generator, self).__init__()
        self.lin=nn.Linear(in_features=hidden_dim, out_features=7*7*256)
        self.norm=nn.BatchNorm1d(num_features=7*7*256)
        self.unflat=nn.Unflatten(1, (256, 7, 7))
        self.convtransp=nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=(5,5), stride=(1,1), padding=2)
        self.norm1=nn.BatchNorm2d(128)
        self.convtransp1 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=(5, 5), stride=(2, 2), padding=2, output_padding=1)
        self.norm2 = nn.BatchNorm2d(64)
        self.convtransp2 = nn.ConvTranspose2d(in_channels=64, out_channels=1, kernel_size=(5, 5), stride=(2, 2), padding=2, output_padding=1)
        self.relu=nn.ReLU()
        self.sigm=nn.Sigmoid()
    def forward(self, x):
        x=self.norm(self.relu(self.lin(x)))
        x=self.unflat(x)
        x=self.norm1(self.relu(self.convtransp(x)))
        x = self.norm2(self.relu(self.convtransp1(x)))
        return self.sigm(self.convtransp2(x))



class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv=nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(5, 5), stride=2, padding=2)
        self.conv1=nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(5, 5), stride=2, padding=2)
        self.flat=nn.Flatten(start_dim=1)
        self.lin=None
        self.lrelu=nn.LeakyReLU()
        self.dropout2d=nn.Dropout2d()

    def forward(self, x):

        x=self.dropout2d(self.lrelu(self.conv(x)))
        x = self.dropout2d(self.lrelu(self.conv1(x)))
        x=self.flat(x)

        if self.lin==None:
            _, in_features = x.shape
            self.lin=nn.Linear(in_features=in_features, out_features=1)
        return self.lin(x)



class GenLoss(nn.Module):
    def __init__(self):
        super(GenLoss, self).__init__()
        self.bce=nn.BCEWithLogitsLoss()
    def forward(self, fake):
        return self.bce(fake, torch.ones_like(fake))

class DisLoss(nn.Module):
    def __init__(self):
        super(DisLoss, self).__init__()
        self.bce=nn.BCEWithLogitsLoss()
    def forward(self, fake, real):
        return self.bce(real, torch.ones_like(real))+self.bce(fake, torch.zeros_like(fake))

def train_step(images, gen_optim, dis_optim, generator, discriminator, hidden_dim, Dis_loss, Gen_loss):

    nums=torch.randn(images.shape[0], hidden_dim)
    gen_images=generator(nums)
    gen_quality=discriminator(gen_images)
    gen_loss=Gen_loss(gen_quality)
    gen_optim.zero_grad()
    gen_loss.backward(retain_graph=True)
    gen_optim.step()


    gen_quality = discriminator(gen_images.detach())
    real_quality = discriminator(images)
    dis_loss=Dis_loss(gen_quality, real_quality)
    dis_optim.zero_grad()
    dis_loss.backward()
    dis_optim.step()

    return gen_loss.item(), dis_loss.item()

def train(dataset, generator, discriminator, hidden_dim, num_epochs, th):
    history=[]
    Dis_loss=DisLoss()
    Gen_loss=GenLoss()
    gen_optim=optim.Adam(generator.parameters(), lr=0.0001)
    dis_optim = optim.Adam(discriminator.parameters(), lr=0.0001)
    for epoch in range(num_epochs):
        print(f"{epoch}/{num_epochs}:", end=" ")
        start_time=time.time()
        n=0
        gen_loss_epoch=0
        for images in dataset():
            gen_loss, dim_loss=train_step(images, gen_optim, dis_optim, generator, discriminator, hidden_dim, Dis_loss, Gen_loss)
            gen_loss_epoch+=gen_loss
            if n%th==0: print("=", end=" ")
            n+=1
        if not n==0:
            history.append(gen_loss_epoch/n)
            print(f": {str(history[-1])}")
            print(f"Время эпохи {epoch} составляет {time.time()-start_time}")
    return history

if __name__=="__main__":
    hidden_dim=2
    generator=Generator(hidden_dim)
    discriminator=Discriminator()
    num=5
    num_epochs=20
    batch_size=100
    dataset=lambda: get_dataset(num, train_dataset, batch_size)
    amount=get_dataset_len(num, train_dataset)
    th=amount//(batch_size*10)
    history=train(dataset, generator, discriminator, hidden_dim, num_epochs, th)
    plt.plot(history)
    plt.show()
    plt.figure()
    with torch.no_grad():
        generator.eval()
        n=5
        k=1
        for i in range(-n, n+1):
            for j in range(-n, n+1):
                ax=plt.subplot(2*n+1, 2*n+1, k)
                ax.imshow(generator(torch.tensor([[1*i/n, 1*j/n]]))[0,0], cmap="gray")
                k+=1
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)
        plt.show()