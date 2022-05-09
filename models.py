import torch.nn as nn
import torch
import torch.nn.functional as F

class CLD3Model(nn.Module):
    def __init__(self, vocab_size,
                        embedding_dim,
                        hidden_dim,
                        label_size,
                        num_ngram_orders):
        super(CLD3Model, self).__init__()
        num_ngram_orders = num_ngram_orders
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.hidden_layer = nn.Linear(embedding_dim*num_ngram_orders, hidden_dim)
        self.softmax_layer = nn.Linear(hidden_dim, label_size)

    def forward(self, ngrams, ngrams_weights):
        import pdb; pdb.set_trace()

        # NGRAM/WEIGHTS INPUT IS
        # BATCH SIZE X NUM ORDERS X MAX LENGTH X NUM HASHES
        weighted_embeddings = self.embedding(ngrams) * ngrams_weights.unsqueeze(-1)
        inputs = torch.sum(weighted_embeddings, dim=2)
        inputs = torch.mean(inputs, dim=2)
        embed = inputs.view(inputs.shape[0], -1)
        hidden = self.hidden_layer(embed)
        output = F.log_softmax(self.softmax_layer(hidden), dim=-1)
        return output
