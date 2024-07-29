import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
import numpy as np
from torch.utils.data import Dataset, DataLoader
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def get_text(text_path):
    with open(text_path, "r", encoding="utf-8") as f:
        text = f.read()
        text = text.replace("\ufeff", "")
        text = re.sub(r"[^А-я ]", "", text)
        text = text.lower()
    return text
def generator(text):
    for c in text:
        yield c


def class_accuracy(y, y_pred):
    return torch.sum((y == torch.max(y_pred, dim=1)[1]).to(torch.int16)) / y.shape[0]


def text_to_one_hot(text):
    gen = generator(text)
    vocab = build_vocab_from_iterator(gen)
    dic = vocab.get_stoi()
    dic = {key: dic[key] + 1 for key in dic}

    one_hot = F.one_hot(torch.tensor(list(dic.values())))
    alphahot = {list(dic.keys())[i]: one_hot[i].tolist() for i in range(len(dic))}
    return [alphahot[c] for c in text], alphahot


class SimpleRNN(nn.Module):
    def __init__(self, num_input, num_hidden):
        super(SimpleRNN, self).__init__()
        self.hidden=num_hidden
        self.rnn=nn.RNN(input_size=num_input, hidden_size=num_hidden, batch_first=True)
        self.lin=nn.Linear(in_features=num_hidden, out_features=num_input)

    def forward(self, x):
        h=torch.zeros(1, x.shape[0], self.hidden).to(device=device)
        output, layer_out=self.rnn(x, h)
        layer_out=layer_out.squeeze(dim=0)

        return self.lin(layer_out)

class TextData(Dataset):
    def __init__(self, X, Y, alphahot):
        self.X=X
        self.Y=Y

    def __getitem__(self, index):

        return torch.tensor(X[index]), torch.argmax(torch.tensor(Y[index]))

    def __len__(self):
        return X.shape[0]


text=get_text("train_data_true.txt")
matr_text, alphahot=text_to_one_hot(text)
matr_text=np.array(matr_text, dtype=np.float32)
inp_chars=7
if __name__=="__main__":

    X=np.array([matr_text[i:i+inp_chars] for i in range(matr_text.shape[0]-inp_chars)])

    Y=matr_text[inp_chars:]

    text_dataset=TextData(X, Y, alphahot)
    text_dataloader=DataLoader(dataset=text_dataset, batch_size=64, shuffle=False)
    model=SimpleRNN(matr_text.shape[1], 256).to(device=device)

    num_epochs=200
    Loss=nn.CrossEntropyLoss()
    optimizer=optim.Adam(model.parameters())
    step_lr_schedular=optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)
    for epoch in range(num_epochs):
        for i, (x, y) in enumerate(text_dataloader):
            x=x.to(device=device)
            y=y.to(device=device)
            optimizer.zero_grad()
            y_pred = model(x)
            loss = Loss(y_pred, y)
            loss.backward()
            optimizer.step()

            print(f"epoch={epoch}, loss={loss}, accuracy={class_accuracy(y.detach(), y_pred.detach())}")
        step_lr_schedular.step()

    y_pred=model(torch.tensor(X))
    y=torch.argmax(torch.tensor(Y), dim=1)
    loss = Loss(y_pred, y)
    print(f"final_loss={loss}, final_accuracy={class_accuracy(y, y_pred.detach())}")
    def get_key_from_value(dictionary, search_value):
        for key, value in dictionary.items():
            if value == search_value:
                return key
        return None
    def build_phrase(inp_str, str_len):
        for i in range(str_len):
            x=[]
            for j in range(i, i+inp_chars):
                x.append(alphahot[inp_str[j]])
            x=torch.tensor(x, dtype=torch.float32).unsqueeze(dim=0).to(device=device)
            c=F.one_hot(torch.max(model(x), dim=1)[1], num_classes=34).squeeze(dim=0).tolist()

            c=get_key_from_value(alphahot, c)
            inp_str+=c

        return inp_str

    torch.save(model.state_dict(), 'alphabet_pred.pth')
    print(build_phrase("утренни", 100))

