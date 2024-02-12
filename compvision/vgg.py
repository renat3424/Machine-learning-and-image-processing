import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from sklearn.datasets import load_digits



def normalization(data, mu, std):
    _data = []
    for x in data:
        _x = x - mu
        for i, __x in enumerate(_x):
            if std[i] == 0:
                _x[i] = 0
            else:
                _x[i] = __x / std[i]
        _data.append(_x)
    return np.array(_data)


def dataset_divide(data, target, train_coef):
    train_count = int(len(data) * train_coef)
    valid_count = int((len(data) - train_count) / 2)
    mixed = []
    for i, x in enumerate(data):
        mixed.append([list(x), target[i]])
    mixed = np.array(mixed)
    np.random.shuffle(mixed)
    train_data, train_t = list(mixed[:train_count, 0]), list(mixed[:train_count, 1])
    valid_data, valid_t = list(mixed[train_count: train_count + valid_count, 0]), list(mixed[
                                                                                       train_count: train_count + valid_count,
                                                                                       1])
    test_data, test_t = list(mixed[train_count + valid_count:, 0]), list(mixed[train_count + valid_count:, 1])

    return train_data, valid_data, test_data, train_t, valid_t, test_t


class VGG(nn.Module):
    def __init__(self, num_classes=10):
        super(VGG, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU())
        self.layer2 = nn.Sequential(
            nn.Conv2d(8, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(8, 10, kernel_size=3, padding=1),
            nn.ReLU())
        self.layer4 = nn.Sequential(
            nn.Conv2d(10, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2))
        self.fc_seq = nn.Sequential(
            nn.Linear(16 * 2 * 2, 100),
            nn.Linear(100, 10)
        )


    def forward(self, x):

        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = out.view(-1, 16 * 2 * 2)

        x = x.view(-1, 16 * 2 * 2)
        return nn.functional.softmax(self.fc_seq(out), dim=1)

    def predict(self, inputs, device='cpu'):
        inputs = inputs.to(device)
        return torch.argmax(self(inputs), dim=1).numpy()





def train(model, optimizer, loss_fn, train_loader, val_loader, epochs=5, device="cpu"):
    for epoch in range(1, epochs + 1):
        training_loss = 0.0
        valid_loss = 0.0
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            inputs, targets = batch
            inputs = inputs.to(device)
            targets = targets.to(device)
            output = model(inputs)
            loss = loss_fn(output, targets)
            loss.backward()
            optimizer.step()
            training_loss += loss.data.item()
        training_loss /= len(train_loader)

        model.eval()
        num_correct = 0
        num_examples = 0
        for batch in val_loader:
            inputs, targets = batch
            inputs = inputs.to(device)
            output = model(inputs)
            targets = targets.to(device)
            loss = loss_fn(output, targets)
            valid_loss += loss.data.item()
            correct = torch.eq(torch.max(F.softmax(output, dim=1), dim=1)[1],
                               targets)
            num_correct += torch.sum(correct).item()
            num_examples += correct.shape[0]
        valid_loss /= len(val_loader)

        print(
            'Epoch: {}, Training Loss: {:.2f}, Validation Loss: {:.2f}, accuracy = {:.2f}'.format(epoch, training_loss,
                                                                                                  valid_loss,
                                                                                                  num_correct / num_examples))

batch_size = 64

# начальные данные
digits = load_digits()
DESCR = digits.DESCR  # описание набора данных
images = digits.images  # массив из 1797 изображений, размер массива 1797х8х8
target = digits.target  # массив из меток изображений, 1797 элементов, значения от 0 до 9
target_names = digits.target_names  # массив имен меток, 10 элементов от 0 до 9
data = digits.data  # массив из “вытянутых” в строку 1797 изображений, размер 1797х64

# нормализация данных разделение выборки
mean = np.mean(data, axis=0)
std = np.std(data, axis=0)
data_n = normalization(data, mean, std)



train_data, valid_data, test_data, train_t, valid_t, test_t = dataset_divide(images, target, 0.8)

# перевод в тензоры
train_data = torch.Tensor(train_data).resize(len(train_data), 1, 8, 8)
print(train_data[0])
valid_data = torch.Tensor(valid_data).resize(len(valid_data), 1, 8, 8)
test_data = torch.Tensor(test_data).resize(len(test_data), 1, 8, 8)

train_t = torch.LongTensor(train_t)
valid_t = torch.LongTensor(valid_t)
test_tt = torch.LongTensor(test_t)

from torch.utils.data import TensorDataset

train_dataset = TensorDataset(train_data, train_t)
valid_dataset = TensorDataset(valid_data, valid_t)
test_dataset = TensorDataset(test_data, test_tt)

# создание загрузчиков

train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)
valid_data_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size)
test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)



# работа нейросети
vgg = VGG()
device = torch.device("cpu")
vgg.to(device)
optimizer = optim.Adam(vgg.parameters(), lr=0.005, weight_decay=0.003)
train(vgg, optimizer, torch.nn.CrossEntropyLoss(), train_data_loader, valid_data_loader, epochs=10, device=device)
# predicted = vgg.predict(test_data_loader)
count = 0
all_count = 0
for input, target in test_data_loader:
    for i in range(len(input)):
        predicted = vgg.predict(input)
        if predicted[i] == target[i]:
            count += 1
        all_count += 1
print(f'Accuracy on test sample {count / all_count}')


