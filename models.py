import torch.nn as nn
import torch
import torch.nn.functional as F
import torchvision
# import json
import math

class CLD3Model(nn.Module):
    def __init__(self, vocab_size,
                        embedding_dim,
                        hidden_dim,
                        label_size,
                        num_ngram_orders):
        super(CLD3Model, self).__init__()
        self.vocab_size = vocab_size
        self.num_ngram_orders = num_ngram_orders
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.label_size = label_size
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.hidden_layer = nn.Linear(embedding_dim*num_ngram_orders, hidden_dim)
        self.softmax_layer = nn.Linear(hidden_dim, label_size)

    def forward(self, inputs):

        # NGRAM/WEIGHTS INPUT IS
        # BATCH SIZE X NUM ORDERS X MAX LENGTH X NUM HASHES
        ngrams = inputs[0]
        weights = inputs[1]

        weighted_embeddings = self.embedding(ngrams) * weights.unsqueeze(-1)
        inputs = torch.sum(weighted_embeddings, dim=2)
        inputs = torch.mean(inputs, dim=2)
        embed = inputs.view(inputs.shape[0], -1)
        hidden = self.hidden_layer(embed)
        output = F.log_softmax(self.softmax_layer(hidden), dim=-1)
        return output

    def save_object(self):
        save = {
            'weights': self.state_dict(),
            'vocab_size': self.vocab_size,
            "num_ngram_orders": self.num_ngram_orders,
            "embedding_dim": self.embedding_dim,
            "hidden_dim": self.hidden_dim,
            "label_size": self.label_size
        }
        return save

class TransformerModel(nn.Module):
    def __init__(self,
                    vocab_size,
                    embedding_dim,
                    label_size,
                    num_layers,
                    max_len,
                    nhead=8
                    ):
        super(TransformerModel, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(embedding_dim, nhead=nhead)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pos_embed = PositionalEncoding(embed_size=embedding_dim, max_len=max_len)
        self.embed = nn.Embedding(vocab_size, embedding_dim)
        self.proj = nn.Linear(embedding_dim, label_size)

        self.vocab_size = vocab_size
        self.embedding_size = embedding_dim
        self.label_size = label_size
        self.num_layers = num_layers
        self.max_length = max_len
        self.nhead = nhead

    def forward(self, inputs): # need to do padding mask
        inputs = inputs[:, :self.max_length]
        pad_mask = self.get_padding_mask(inputs)
        inputs = inputs.t()
        embed = self.embed(inputs)
        pos_embed = self.pos_embed(embed)
        encoding = torch.mean(self.encoder(pos_embed, src_key_padding_mask=pad_mask), dim=0)
        output = self.proj(encoding)
        return F.log_softmax(output, dim=-1)

    def get_padding_mask(self, inputs):
        return torch.eq(inputs, 0)

    def save_object(self):
        save = {
            "weights": self.state_dict(),
            "vocab_size": self.vocab_size,
            "embedding_size": self.embedding_size,
            "label_size": self.label_size,
            "num_layers": self.num_layers,
            "max_length": self.max_length,
            "nhead": self.nhead
        }
        return save

class PositionalEncoding(nn.Module):
    
    def __init__(self, embed_size=512, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, embed_size)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_size, 2).float() * (-math.log(10000.0) / embed_size))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0,1)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class ConvModel(nn.Module):
    def __init__(self,
                    vocab_size,
                    label_size,
                    embedding_dim=256,
                    conv_min_width=2,
                    conv_max_width=5,
                    conv_depth=64):
        super(ConvModel, self).__init__()
        self.vocab_size = vocab_size
        self.label_size = label_size
        self.embedding_dim = embedding_dim
        self.conv_min_width=conv_min_width
        self.conv_max_width=conv_max_width
        self.conv_depth=conv_depth

        self.embed = nn.Embedding(vocab_size, embedding_dim)

        self.convolutions = nn.ModuleList([
            nn.Conv1d(in_channels=embedding_dim, out_channels=embedding_dim, kernel_size=width) for width in range(conv_min_width, conv_max_width)
        ])

        self.proj = nn.Linear(embedding_dim, label_size)

    def forward(self, inputs):
        embed = self.embed(inputs).transpose(1,2)
        features = []
        for layer in self.convolutions:
            z = layer(embed)
            features.append(z)
        
        features = torch.cat(features, dim=2)
        mean = torch.mean(features, dim=2)
        output = self.proj(mean)
        return F.log_softmax(output, dim=-1)

    def save_object(self):
        save = {
            "weights": self.state_dict(),
            "vocab_size": self.vocab_size,
            "embedding_size": self.embedding_dim,
            "label_size": self.label_size,
            "conv_min_width": self.conv_min_width,
            "conv_max_width": self.conv_max_width,
        }
        return save


class Block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=in_ch, 
                                out_channels=out_ch,
                                kernel_size=3)
        self.relu  = nn.ReLU()
        self.conv2 = nn.Conv1d(in_channels=out_ch, 
                                out_channels=out_ch,
                                kernel_size=3)
    
    def forward(self, x):
        return self.relu(self.conv2(self.relu(self.conv1(x))))

class Encoder(nn.Module):
    def __init__(self, chs=(1,2,4,8,16,32)):
        super().__init__()
        self.enc_blocks = nn.ModuleList([Block(chs[i], chs[i+1]) for i in range(len(chs)-1)])
        self.pool       = nn.MaxPool1d(2)
    
    def forward(self, x):
        ftrs = []
        for block in self.enc_blocks:
            x = block(x)
            ftrs.append(x)
            x = self.pool(x)
        return ftrs

class Decoder(nn.Module):
    def __init__(self, chs=(32, 16, 8, 4, 2)):
        super().__init__()
        self.chs         = chs
        self.upconvs    = nn.ModuleList([nn.ConvTranspose1d(chs[i], chs[i+1], 2) for i in range(len(chs)-1)])
        self.dec_blocks = nn.ModuleList([Block(chs[i], chs[i+1]) for i in range(len(chs)-1)]) 
        
    def forward(self, x, encoder_features):
        for i in range(len(self.chs)-1):
            x        = self.upconvs[i](x)
            enc_ftrs = self.crop(encoder_features[i], x)
            x        = torch.cat([x, enc_ftrs], dim=1)
            x        = self.dec_blocks[i](x)
        return x
    
    def crop(self, enc_ftrs, x):
        _, C, L = x.shape
        enc_ftrs   = torchvision.transforms.CenterCrop([C,L])(enc_ftrs)
        return enc_ftrs

class UNETModel(nn.Module):
    def __init__(self,
                    vocab_size,
                    label_size,
                    embed_size=128,
                    length=1024):
        super(UNETModel, self).__init__()
        self.chs = [embed_size*(2**i) for i in range(5)]

        self.embed_size = embed_size
        self.length = length

        self.embed = nn.Embedding(vocab_size, embed_size)
        self.encoder = Encoder(chs=self.chs)
        self.decoder = Decoder(chs=self.chs[::-1][:-1])
        self.proj = nn.Linear(self.chs[1], label_size)

    
    def forward(self, inputs):
        emb = self.embed(inputs).transpose(1,2)
        encoding = self.encoder(emb)
        decoding = self.decoder(encoding[::-1][0], encoding[::-1][1:])
        upsampled = F.interpolate(decoding, self.length)
        out = self.proj(upsampled.transpose(1,2))

        return F.log_softmax(out, dim=2)

    def save_object(self):
        save = {}
        return save
