import numpy as np
import re
import torch
from torchtext.vocab import build_vocab_from_iterator
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from torchsummary import summary
import torch.optim as optim

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def preprocess_text(text):
    text=re.sub("[^А-я ]", "", text)
    text=text.lower()
    return text
def get_lines(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        lines=f.readlines()
        lines[0].replace("\ufeff", "")
        lines=list(map(preprocess_text, lines))

    return lines

def generate(lines):
    for line in lines:
        for word in line.split(" "):
            if not word=="":
                yield word

def lines_to_numbers(lines, vocab):
    new_lines=[]
    for line in lines:
        lst=[]
        for word in line.split(" "):
            if not word == "" and word in vocab:
                lst.append(vocab[word])
        new_lines.append(lst)
    return new_lines


def pad_line(line, max_number):
    if len(line)<max_number:
        return (max_number-len(line))*[0]+line
    else:
        return line[:max_number]



def accuracy(y_pred, y):
    return ((y==torch.argmax(y_pred, dim=1)).sum()/y.shape[0]).item()
class PhrasesDataset(Dataset):
    def __init__(self, X, Y):
        self.X=X
        self.Y=Y

    def __getitem__(self, index):
        return self.X[index], self.Y[index]

    def __len__(self):
        return self.Y.shape[0]



class SentimentAnalysis(nn.Module):
    def __init__(self, number_of_words, input_length, number_of_layers):
        self.number_of_layers=number_of_layers
        super(SentimentAnalysis, self).__init__()
        self.emb=nn.Embedding(num_embeddings=number_of_words, embedding_dim=input_length)
        self.lstm=nn.GRU(input_size=input_length, num_layers=number_of_layers, hidden_size=128, batch_first=True)
        self.lstm1=nn.GRU(input_size=128, num_layers=number_of_layers, hidden_size=64, batch_first=True)
        self.lin=nn.Linear(in_features=64, out_features=2)


    def forward(self, x:torch.Tensor):
        N,S=x.shape
        h0=torch.zeros(self.number_of_layers, N, 128)
        c0 = torch.zeros(self.number_of_layers, N, 128)
        x=self.emb(x)
        x, _=self.lstm(x, h0)
        h1 = torch.zeros(self.number_of_layers, N, 64)
        c1 = torch.zeros(self.number_of_layers, N, 64)
        x, h1 = self.lstm1(x, h1)
        return self.lin(h1[-1])



def train(model, Loss, optimizer, dataloader, num_epochs, X, Y):
    X=X.to(device=device)
    Y=Y.to(device=device)
    for epoch in range(num_epochs):
        for i, (x, y) in enumerate(dataloader):
            x=x.to(device=device)
            y = y.to(device=device)
            optimizer.zero_grad()
            y_pred=model(x)
            loss=Loss(y_pred, y)
            loss.backward()
            optimizer.step()
            print(f"loss={loss}, accuracy={accuracy(y_pred, y)}")
        with torch.no_grad():
            y_pred = model(X)
            print(f"epoch={epoch}, epoch_accuracy={accuracy(y_pred, Y)}")
    torch.save(model.state_dict(), "sentiment.pth")

true_lines=get_lines("train_data_true.txt")
false_lines=get_lines("train_data_false.txt")
true_lines_len=len(true_lines)
false_lines_len=len(false_lines)
all_lines=true_lines+false_lines
all_lines_len=len(all_lines)
vocab=build_vocab_from_iterator([generate(all_lines)], max_tokens=1000)

vocab={key: value+1 for key, value in vocab.get_stoi().items()}
reverse_vocab=dict(map(reversed, vocab.items()))
X=lines_to_numbers(all_lines, vocab)
max_text_len=10
X=torch.tensor(list(map(lambda x: pad_line(x, max_text_len), X)))
Y=torch.tensor([1]*true_lines_len+[0]*false_lines_len)

if __name__=="__main__":


    phrases_dataset=PhrasesDataset(X, Y)
    dataloader=DataLoader(dataset=phrases_dataset, shuffle=True, batch_size=32)

    number_of_words=len(vocab)+1
    model=SentimentAnalysis(number_of_words, 128, 1).to(device=device)
    Loss=nn.CrossEntropyLoss()
    optimizer=optim.Adam(model.parameters(), 0.0001)
    num_epochs=50
    train(model, Loss, optimizer, dataloader, num_epochs, X, Y)



