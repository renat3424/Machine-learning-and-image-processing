import torch
import torch.nn as nn
import torchvision
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

class MnDataSet(Dataset):
    def __init__(self):
        data=torchvision.datasets.MNIST("./", transform=torchvision.transforms.ToTensor)
        self.x=data.data.to(torch.float32)/255
        self.x=self.x.unsqueeze(1)
        self.y=data.targets
        self.n_samples=self.y.shape[0]
        self.output_number = len(data.classes)

    def __getitem__(self, index):
        return self.x[index], self.y[index]


    def __len__(self):
        return self.n_samples




class RecCNN(nn.Module):
    def __init__(self, n_output, in_channels=1):

        super(RecCNN, self).__init__()
        self.conv1=nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=(3,3), padding="same")
        self.relu=nn.ReLU()
        self.pooling=nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2=nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3,3), padding="same")
        self.flatten=nn.Flatten(start_dim=1)
        self.lin1=nn.Linear(in_features=7*7*64, out_features=128)
        self.lin2=nn.Linear(in_features=128, out_features=n_output)

    def forward(self, x):
        out=self.conv1(x)
        out=self.relu(out)
        out=self.pooling(out)
        out = self.conv2(out)
        out = self.relu(out)
        out = self.pooling(out)
        out=self.flatten(out)
        out=self.lin1(out)
        out = self.relu(out)
        return self.lin2(out)

def accuracy(y_pred, y_real):
    return torch.sum((torch.max(y_pred, dim=1)[1]==y_real).to(torch.int16))/y_real.shape[0]

if __name__=="__main__":
    train_dataset=MnDataSet()
    batch_size=100
    train_dataloader=DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_dataset=torchvision.datasets.MNIST("./", transform=torchvision.transforms.ToTensor, train=False)
    test_data=(test_dataset.data.to(torch.float32)/255).unsqueeze(1)
    test_target=test_dataset.targets

    model=RecCNN(train_dataset.output_number)
    Loss=nn.CrossEntropyLoss()
    optimizer=optim.Adam(model.parameters(), lr=0.003)
    num_epochs=10
    losses=[]
    for epoch in range(num_epochs):
        for i, (x, y) in enumerate(train_dataloader):
            y_pred=model(x)
            loss=Loss(y_pred, y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            losses.append(loss.item())
            print(f"epoch {epoch + 1}/{num_epochs}, step {i + 1}/{int(train_dataset.n_samples / batch_size)}, loss={loss.item()}, accuracy={accuracy(y_pred, y)}")

        with torch.no_grad():
            y_pred = model(test_data).detach()
            loss = Loss(y_pred, test_target)

            print(f"loss and accuracy for epoch={epoch}, loss={loss.item()}, accuracy={accuracy(y_pred, test_target)}")

