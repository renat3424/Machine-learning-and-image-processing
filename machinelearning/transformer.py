import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    def __init__(self, embed, heads):
        super(SelfAttention, self).__init__()
        self.embed=embed
        self.heads=heads
        self.head_size=self.embed//self.heads
        assert (self.head_size*self.heads==self.embed), "Heads amount have to divide embedding size"
        self.values=nn.Linear(self.head_size, self.head_size, bias=False)
        self.keys = nn.Linear(self.head_size, self.head_size, bias=False)
        self.queries = nn.Linear(self.head_size, self.head_size, bias=False)
        self.fc_out=nn.Linear(self.head_size*self.heads, self.embed)

    def forward(self, values, keys, queries, mask):
        N=queries.shape[0]
        value_length, key_length, query_length=values.shape[1], keys.shape[1], queries.shape[1]
        values=values.reshape(N, value_length, self.heads, self.head_size)
        keys = keys.reshape(N, key_length, self.heads, self.head_size)
        queries = queries.reshape(N, query_length, self.heads, self.head_size)

        values=self.values(values)
        keys = self.keys(keys)
        queries= self.queries(queries)

        energy=torch.einsum("bqsh,bksh->bsqk", [queries, keys])

        if not mask==None:
            energy=energy.masked_fill(mask==0, float(-1e20))

        energy=torch.softmax(energy/(self.head_size**(1/2)), dim=3)
        out=torch.einsum("bsqk,bksh->bqsh", [energy, values]).reshape(N, query_length, self.heads*self.head_size)
        return self.fc_out(out)


class TransformerBlock(nn.Module):
    def __init__(self, embed, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        self.attention=SelfAttention(embed, heads)
        self.layer_norm=nn.LayerNorm(embed)
        self.ff=nn.Sequential(nn.Linear(in_features=embed, out_features=forward_expansion*embed),
                              nn.ReLU(),
                              nn.Linear(in_features=forward_expansion*embed, out_features=embed))
        self.layer_norm1 = nn.LayerNorm(embed)
        self.dropout=nn.Dropout(dropout)

    def forward(self, value, query, key, mask):
        attention=self.attention(value, key, query, mask)
        x=self.dropout(self.layer_norm(attention+query))
        x = self.dropout(self.layer_norm1(self.ff(x) + x))
        return x

class Encoder(nn.Module):
    def __init__(self, src_vocab_size, embed, num_layers, heads, device, forward_expansion, dropout, max_length):
        super(Encoder, self).__init__()
        self.embed=embed
        self.device=device

        self.word_embed=nn.Embedding(src_vocab_size, embed)
        self.position_embed=nn.Embedding(max_length, embed)

        self.layers=nn.ModuleList(
            [
                TransformerBlock(embed, heads, dropout, forward_expansion) for _ in range(num_layers)
            ]
        )
        self.dropout=nn.Dropout(p=dropout)

    def forward(self,x, mask):
        N, seq_size=x.shape
        positions=torch.arange(0, seq_size).expand(N, seq_size).to(device=self.device)
        x=self.dropout(self.word_embed(x)+self.position_embed(positions))
        for layer in self.layers:
            x=layer(x, x, x, mask)
        return x

class DecoderBlock(nn.Module):
    def __init__(self, embed, heads, dropout, forward_expansion):
        super(DecoderBlock, self).__init__()
        self.mask_attention=SelfAttention(embed, heads)
        self.lnorm=nn.LayerNorm(embed)
        self.tr_block=TransformerBlock(embed, heads, dropout, forward_expansion)
        self.dropout=nn.Dropout(p=dropout)

    def forward(self, query, values, keys, src_mask, trg_msk):
        m_att=self.mask_attention(query, query, query, trg_msk)
        query=self.dropout(self.lnorm(m_att+query))
        out=self.tr_block(values, query, keys, src_mask)
        return out

class Decoder(nn.Module):
    def __init__(self, trg_vocab_size, embed, num_layers, heads, device, forward_expansion, dropout, max_length):
        super(Decoder, self).__init__()
        self.device=device

        self.word_embed=nn.Embedding(trg_vocab_size, embed)
        self.position_embed=nn.Embedding(max_length, embed)

        self.layers=nn.ModuleList(
            [
                DecoderBlock(embed, heads, dropout, forward_expansion) for _ in range(num_layers)
            ]
        )
        self.dropout=nn.Dropout(p=dropout)
        self.fc_out=nn.Linear(embed, trg_vocab_size)

    def forward(self, x, enc_out, src_mask, trg_mask):
        N, seq_size=x.shape
        positions=torch.arange(0, seq_size).expand(N, seq_size).to(device=self.device)
        x=self.dropout(self.word_embed(x)+self.position_embed(positions))
        for layer in self.layers:
            x=layer(x, enc_out, enc_out, src_mask, trg_mask)
        return self.fc_out(x)

class Transformer(nn.Module):
    def __init__(self, src_vocab_size, trg_vocab_size, src_pad_idx, trg_pad_idx, embed_size=256, num_layers=6, forward_expansion=4, heads=8, dropout=0, device=torch.device("cuda"), max_length=100):
        super(Transformer, self).__init__()
        self.encoder=Encoder(src_vocab_size, embed_size, num_layers, heads, device, forward_expansion, dropout, max_length)
        self.decoder = Decoder(trg_vocab_size, embed_size, num_layers, heads, device, forward_expansion, dropout,
                               max_length)
        self.src_pad_idx=src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device=device

    def src_mask(self, src):
        return (src!=self.src_pad_idx).unsqueeze(1).unsqueeze(2).to(device=self.device)

    def trg_mask(self, trg):
        N, seq_len=trg.shape
        return torch.tril(torch.ones(seq_len, seq_len)).expand(N, 1, seq_len, seq_len).to(device=self.device)

    def forward(self, src, trg):
        trg_mask=self.trg_mask(trg)
        src_mask = self.src_mask(src)
        enc_src=self.encoder(src, src_mask)
        return self.decoder(trg, enc_src, src_mask, trg_mask)

if __name__=="__main__":

    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # x=torch.tensor([[1,5,6,8,3,4,7,2,0], [1,3,6,9,3,4,7,5,2]]).to(device=device)
    # y=torch.tensor([[1,7,4,3,5,9,2,0], [1,5,6,2,4,7,6,2]]).to(device=device)
    # src_pad_idx=0
    # trg_pad_idx=0
    # src_vocab_size=10
    # trg_vocab_size = 10
    # model=Transformer(src_vocab_size, trg_vocab_size, src_pad_idx,trg_pad_idx, device=device).to(device=device)
    # print(model(x, y[:,:-1]).shape)
