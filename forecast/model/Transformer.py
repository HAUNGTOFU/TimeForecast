import math
import torch
from torch import nn
class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model
    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)
#%%
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=20):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    def forward(self, x):
        x = x + self.pe[:, :x.size(1)].clone().detach()
        return self.dropout(x)
#%%
class Transformer(nn.Module):
    def __init__(self, input_dim,seq_len,fore_len,output_dim, d_model, nhead, num_layers, dropout=0.1, max_len=20):
        super(Transformer,self).__init__()
        self.fc1 = nn.Linear(input_dim,d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout,batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.fc_out1 = nn.Linear(seq_len*d_model, d_model)
        self.fc2 = nn.Linear(d_model,output_dim*fore_len)
    def forward(self, src):
        src = self.fc1(src)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        output = output.flatten(start_dim=1)
        output = self.fc_out1(output)
        output = self.fc2(output)
        return output