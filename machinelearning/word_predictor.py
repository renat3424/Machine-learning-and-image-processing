import numpy as np
import torch
import torch.nn as nn
from torchtext.vocab import build_vocab_from_iterator
from torchtext.data.utils import get_tokenizer
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import re

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')






class WordPredictor(nn.Module):
    def __init__(self, num_inputs, num_hidden, num_layers, num_outputs):

        super(WordPredictor, self).__init__()

        self.num_layers=num_layers
        self.hidden_size = num_hidden
        self.rnn=nn.RNN(input_size=num_inputs, hidden_size=num_hidden, num_layers=num_layers, batch_first=True)
        self.lin=nn.Linear(in_features=num_hidden, out_features=num_outputs)


    def forward(self, x):

        batch_size, *_=x.shape

        h_0 = torch.zeros(self.num_layers, batch_size, self.hidden_size)

        output, layers=self.rnn(x, h_0)
        return self.lin(layers[-1])


class TextDataset(Dataset):
    def get_list_of_words(self, text):
        tokenizer = get_tokenizer(None)
        return tokenizer(text)

    def get_text(self, text_path):
        with open(text_path, "r", encoding="utf-8") as f:
            text = f.read()
            text = re.sub("[\n]", " ", text)
            text = re.sub("[^А-я ]", "", text)
            text = text.lower()
        return text

    def generator(self):
        for word in self.text:
            yield word

    def __init__(self, text, max_words, num_words):
        self.text=self.get_list_of_words(self.get_text(text))
        vocab = build_vocab_from_iterator([self.generator()], max_tokens=max_words)
        self.vocab = {key: value + 1 for key, value in vocab.get_stoi().items()}
        self.num_input=max(self.vocab.values())+1
        self.num_words=num_words

    def __getitem__(self, index):
        x=self.text[index:index+self.num_words]
        x=[self.vocab[word] for word in x]
        y=self.vocab[self.text[index+self.num_words]]
        return F.one_hot(torch.tensor(x), self.num_input).to(dtype=torch.float32),torch.tensor(y)

    def __len__(self):
        return len(self.text)-self.num_words


def train(model, Loss, optimizer, dataloader, device, lr_sch, num_epochs, file_name):
    for epoch in range(num_epochs):
        for i, (x, y) in enumerate(dataloader):

            x=x.to(device=device)
            y = y.to(device=device)
            optimizer.zero_grad()
            y_pred=model(x)
            loss=Loss(y_pred, y)
            loss.backward()
            optimizer.step()
            print(f"epoch={epoch}, lr={optimizer.param_groups[0]['lr']}, loss={loss}, accuracy={class_accuracy(y_pred.detach(), y.detach())}")
        lr_sch.step()
        # with torch.no_grad():
        #     size_prev=dataloader.batch_size
        #     dataloader.len(dataloader.dataset)
        #     x, y = iter(dataloader).__next__()
        #     print(y.shape)
        #     x = x.to(device=device)
        #     y = y.to(device=device)
        #     y_pred = model(x)
        #     dataloader.batch_size = size_prev
        #     print(f"final accuracy for epoch {epoch}={class_accuracy(y_pred, y)}")
    torch.save(model.state_dict(), file_name)




def class_accuracy(y_pred, y):
    return (y==torch.argmax(y_pred, dim=1)).sum()/y.shape[0]


def write_words(input_words, num_words, model, vocab, input_number):
    input_words=input_words.split(sep=" ")
    for i in range(num_words):
        with torch.no_grad():
            print(input_words[i:i+3])
            x=F.one_hot(torch.tensor([vocab[word] for word in input_words[i:i+3]]).unsqueeze(dim=0), input_number).to(device=device, dtype=torch.float32)
            y=torch.argmax(model(x), dim=1).squeeze(dim=0).item()
            word=get_key_from_value(vocab, y)
            input_words.append(word)
    return "".join([word+" " for word in input_words])



def get_key_from_value(dictionary, search_value):
    for key, value in dictionary.items():
        if value == search_value:
            return key
    return None


if __name__=="__main__":
    num_words=3
    dataset = TextDataset("train_data_false.txt", 1000, num_words)
    model = WordPredictor(dataset.num_input, 400, 2, dataset.num_input).to(device=device)
    dataloader=DataLoader(dataset=dataset, batch_size=32, shuffle=False)

    vocab=dataset.vocab
    Loss=nn.CrossEntropyLoss()
    optimizer=optim.Adam(model.parameters())
    learning_step=optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=500, gamma=0.1)
    train(model, Loss, optimizer, dataloader, device, lr_sch=learning_step, num_epochs=1000, file_name="word_predictor_weights.pth")
    print(write_words("этих жизненных препятствий", 1000, model, vocab, dataset.num_input))






