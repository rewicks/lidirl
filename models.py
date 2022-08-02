import torch.nn as nn
import torch
import torch.nn.functional as F
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

        self.convolutions = [
            nn.Conv1d(in_channels=embedding_dim, out_channels=embedding_dim, kernel_size=width) for width in range(conv_min_width, conv_max_width)
        ]

        self.proj = nn.Linear(embedding_dim, label_size)

    def forward(self, inputs):
        embed = self.embed(inputs).transpose(1,2)
        features = []
        for layer in self.convolutions:
            z = layer(embed)
            features.append(z.squeeze())
        
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