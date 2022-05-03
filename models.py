import torch.nn as nn
import torch
import torch.nn.functional as F

class CLD3Model(nn.Module):
    def __init__(self, vocab_size, embedding_dim, label_size, max_ngram_order):
        super(CLD3Model, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.hidden_layer = nn.Linear(embedding_dim*max_ngram_order, embedding_dim)
        self.softmax_layer = nn.Linear(embedding_dim, label_size)

    def forward(self, ngrams, ngrams_weights):
        out = torch.sum(self.embedding(ngrams) * ngrams_weights.view(ngrams_weights.shape[0], ngrams_weights.shape[1], ngrams_weights.shape[2], 1), dim=2)
        embed = out.view(out.shape[0], -1)
        hidden = self.hidden_layer(embed)
        output = F.log_softmax(self.softmax_layer(hidden), dim=-1)
        return output
