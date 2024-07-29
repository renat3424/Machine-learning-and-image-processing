import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
from torchsummary import summary
class Predictor(nn.Module):

    def __init__(self, in_features, hidden_features):
        super(Predictor, self).__init__()
        self.lin=nn.Linear(in_features=in_features, out_features=hidden_features)
        self.norm=nn.BatchNorm1d(num_features=hidden_features)
        self.norm1 = nn.BatchNorm1d(num_features=hidden_features)
        self.lin1 = nn.Linear(in_features=hidden_features, out_features=hidden_features)
        self.lin2=nn.Linear(in_features=hidden_features, out_features=hidden_features)
        self.lin3 = nn.Linear(in_features=hidden_features, out_features=1)
        self.tanh=nn.Tanh()
        self.sigm=nn.Sigmoid()
        self.relu=nn.ReLU()

    def forward(self, x):

        return self.lin3(self.relu(self.norm1(self.lin2(self.relu(self.norm(self.lin1(self.relu(self.lin(x)))))))))


class Predictor1(nn.Module):

    def __init__(self, in_features, hidden_features):
        super(Predictor1, self).__init__()
        self.lin1=nn.Linear(in_features=in_features, out_features=hidden_features)
        self.norm = nn.BatchNorm1d(num_features=hidden_features)
        self.norm1=nn.BatchNorm1d(num_features=hidden_features)
        self.norm2 = nn.BatchNorm1d(num_features=hidden_features)
        self.lin2 = nn.Linear(in_features=hidden_features, out_features=hidden_features)
        self.lin3 = nn.Linear(in_features=hidden_features, out_features=hidden_features)
        self.lin4=nn.Linear(in_features=hidden_features, out_features=256)
        self.relu=nn.Sigmoid()

    def forward(self, x):
        return self.lin4(self.relu(self.norm(self.lin1(x))))


def accuracy(real, pred):
    return (real==pred).sum()/real.shape[0]

def train(model, indexes, numbers):

    Loss=nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.1)
    step_lr_schedular = optim.lr_scheduler.StepLR(optimizer, step_size=30000, gamma=0.1)

    num_epochs = 60000

    for i in range(num_epochs):
        optimizer.zero_grad()
        numbers_pred = model(indexes)
        loss = Loss(numbers_pred, numbers)
        loss.backward()
        optimizer.step()
        step_lr_schedular.step()
        print(f"index={i}, lr={optimizer.param_groups[0]['lr']}, loss={loss}")


num_val=100
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
numbers=np.random.randint(0, 255, num_val)
numbers=torch.tensor(numbers, dtype=torch.int64)



indexes=torch.tensor(np.array([i for i in range(num_val)]).reshape(numbers.shape[0], 1), dtype=torch.float32)

numbers = numbers.to(device=device)
indexes = indexes.to(device=device)
model=Predictor1(1, 6).to(device=device)
print(summary(model.cuda(), (1, 1)))
train(model, indexes, numbers)
numbers_pred=model(indexes)
numbers_pred=torch.max(numbers_pred, dim=1)[1]
print(numbers, numbers_pred)
print(accuracy(numbers, numbers_pred).item())






