import torch.nn as nn
import torch
import torch.nn.functional as F
import json

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

    def forward(self, ngrams, ngrams_weights):

        # NGRAM/WEIGHTS INPUT IS
        # BATCH SIZE X NUM ORDERS X MAX LENGTH X NUM HASHES
        weighted_embeddings = self.embedding(ngrams) * ngrams_weights.unsqueeze(-1)
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
