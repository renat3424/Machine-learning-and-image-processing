import torch
import torch.nn as nn
import torch.optim as optim
import spacy
from torchtext.datasets import Multi30k
from torchtext.data import Field, BucketIterator
from torchtext.data.metrics import bleu_score
from torch.utils.tensorboard import SummaryWriter
from utils import save_checkpoint, load_checkpoint
import numpy as np





spacy_de=spacy.load("de_core_news_sm")
spacy_en=spacy.load("en_core_web_sm")

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
def tokenizer(text, sp):
    return [tol.text for tol in sp.tokenizer(text)]


def index_to_word(word, vocab):
    return vocab.vocab.itos[word]
def bleu(data, model, english, german, device):
    print("data_preparation")
    targets=[]
    predicted=[]
    i=0
    for example in data:

        src=vars(example)["src"]

        trg= vars(example)["trg"]
        print("real:", list_to_sent(trg))
        translated=translate_sentence(model, src, english, german, device)
        translated=translated[:-1]
        print("translated:", list_to_sent(translated))
        predicted.append(translated)
        targets.append([trg])
        i+=1
        if i==100:
            break

    return bleu_score(predicted, targets)

def list_to_sent(lst):
    return "".join(
        [lst[i] + " " if (lst[i + 1] != "," and lst[i + 1] != ".") else lst[i] for i in
         range(len(lst) - 1)] + ["."])



def translate_sentence(model, text, english, german, device, max_length=100):
    sp=spacy.load("de_core_news_sm")
    if type(text)=="str":

        sentence=[tol.text.lower() for tol in sp.tokenizer(text)]
    else:
        sentence = [word.lower() for word in text]

    sentence.insert(0, german.init_token)
    sentence.append(german.eos_token)
    sentence=[german.vocab.stoi[word] for word in sentence]
    start=[english.vocab.stoi[english.init_token]]
    for i in range(max_length):
        out=model(torch.tensor(sentence).unsqueeze(0).to(device=device), torch.tensor(start).unsqueeze(0).to(device=device))
        res=out.argmax(2)[:, -1].item()
        start.append(res)
        if res==english.vocab.stoi[english.eos_token]:
            break
    return [english.vocab.itos[word] for word in start[1:]]





german=Field(tokenize=lambda text: tokenizer(text, spacy_de), lower=True, init_token='<sos>', eos_token='<eos>', batch_first=True)

english=Field(tokenize=lambda text: tokenizer(text, spacy_en), lower=True, init_token='<sos>', eos_token='<eos>', batch_first=True)

train_data, validation_data, test_data=Multi30k.splits(exts=(".de", ".en"), fields=(german, english))



german.build_vocab(train_data, max_size=10000, min_freq=2)
english.build_vocab(train_data, max_size=10000, min_freq=2)






class Transformer(nn.Module):
    def __init__(self, embed_size, num_heads, src_pad_idx, num_encoder_layers, num_decoder_layers, forward_expansion, dropout, src_vocab_size, trg_vocab_size, max_len, device):
        super(Transformer, self).__init__()
        self.src_embed=nn.Embedding(src_vocab_size, embedding_dim=embed_size)
        self.trg_embed = nn.Embedding(trg_vocab_size, embedding_dim=embed_size)

        self.add=nn.Linear(in_features=embed_size, out_features=4*embed_size)
        self.add1=nn.Linear(in_features=4*embed_size, out_features=embed_size)
        self.batch_norm=nn.BatchNorm1d(4*embed_size)

        self.src_pos_embed = nn.Embedding(max_len, embedding_dim=embed_size)
        self.trg_pos_embed = nn.Embedding(max_len, embedding_dim=embed_size)
        self.transformer=nn.Transformer(d_model=embed_size, nhead=num_heads, num_encoder_layers=num_encoder_layers,
                                        num_decoder_layers=num_decoder_layers,
                                        dim_feedforward=forward_expansion*embed_size, batch_first=True, dropout=dropout)
        self.fc_out=nn.Linear(in_features=embed_size, out_features=trg_vocab_size)
        self.device=device
        self.dropout=nn.Dropout(p=dropout)
        self.src_pad_idx=src_pad_idx

    def create_src_mask(self, src)->torch.Tensor:
        return src==self.src_pad_idx

    def forward(self, src, trg):
        src_mask=self.create_src_mask(src).to(device=self.device)
        trg_mask=self.transformer.generate_square_subsequent_mask(trg.shape[1]).to(device=self.device)
        N, src_len=src.shape
        N, trg_len = trg.shape
        pos1=torch.arange(0, src_len).unsqueeze(0).expand(N, src_len).to(device=device)
        pos2 = torch.arange(0, trg_len).unsqueeze(0).expand(N, trg_len).to(device=device)
        src=self.dropout(self.src_embed(src)+self.src_pos_embed(pos1))
        src=self.add(src)
        src=self.batch_norm(src.permute(0,2,1)).permute(0,2,1)
        src=self.add1(src)
        trg = self.dropout(self.trg_embed(trg) + self.trg_pos_embed(pos2))
        return self.fc_out(self.transformer(src, trg, src_key_padding_mask=src_mask, tgt_mask=trg_mask))




def train(model, optimizer, Loss, data_iter, num_epochs, german_example, eng_example, english, german, writer, lr_scheduler):
    step=0
    for epoch in range(num_epochs):
        checkpoint={"state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict()}
        save_checkpoint(checkpoint)
        model.eval()

        print(f"epoch={epoch}/{num_epochs}")
        translated=translate_sentence(model, german_example, english, german, device=device)
        print(f"translated sentence: {list_to_sent(translated)}" )
        print(f"real sentence: {list_to_sent(eng_example)}")
        model.train()
        for i, batch in enumerate(data_iter):
            src=batch.src
            trg=batch.trg
            pred=model(src, trg)[:,:-1,:]

            pred=pred.reshape(-1, pred.shape[2])
            trg=trg[:, 1:].reshape(-1)
            optimizer.zero_grad()
            loss=Loss(pred, trg)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
            optimizer.step()

            print(f"i={i}, loss={loss.item()}, optim_lr={optimizer.param_groups[0]['lr']}")
            writer.add_scalar("Training loss", loss, global_step=step)
            step+=1
        lr_scheduler.step()






example=test_data[0].__dict__
ger_example=example["src"]
eng_example=example["trg"]
num_epochs=20
learning_rate=3e-4
batch_size=32

src_vocab_size=len(german.vocab)
trg_vocab_size=len(english.vocab)
embedding_size=512
num_heads=8
num_encoder_layers=3
num_decoder_layers=3
dropout=0.1
max_len=100
forward_expansion=4
src_pad_idx=german.vocab.stoi["<pad>"]
writer=SummaryWriter("runs/loss_plot")
step=0
train_iter, val_iter, test_iter=BucketIterator.splits((train_data, validation_data, test_data), batch_size=batch_size, sort_within_batch=True, sort_key=lambda x: len(x.src), device=device)
model=Transformer(embed_size=embedding_size, num_heads=num_heads, src_pad_idx=src_pad_idx,
                  num_encoder_layers=num_encoder_layers, num_decoder_layers=num_decoder_layers,
                  forward_expansion=forward_expansion, dropout=dropout, src_vocab_size=src_vocab_size,
                  trg_vocab_size=trg_vocab_size, max_len=max_len, device=device).to(device=device)


optimizer=optim.Adam(model.parameters(), lr=learning_rate)
lr_scheduler=optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
Loss=nn.CrossEntropyLoss()

train(model, optimizer, Loss, train_iter, num_epochs, ger_example, eng_example, english, german, writer, lr_scheduler)
model.eval()
score=bleu(test_data, model, english, german, device)
print(score)
