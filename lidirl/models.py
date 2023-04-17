#!/usr/bin/env python3

"""
    Holds different types of models that we've experimented with
"""

################################ PACKAGING AND LOGGING ################################
import pathlib
import logging
import os, sys

if __package__ is None and __name__ == '__main__':
    parent = pathlib.Path(__file__).absolute().parents[1]
    sys.path.insert(0, str(parent))
    __package__ = 'lidirl'


logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stderr,
)
logger = logging.getLogger("lidirl")

#################################### FUNCTIONALITY ####################################
import math

import roformer
import torch, torchvision
import torch.nn as nn
import torch.nn.functional as F
from transformers import RoFormerModel, RoFormerConfig
from torch.types import _dtype as DType

from . import __version__
from .MCLayer import MCSoftmaxDenseFA



######################################## CLD3 ########################################


class CLD3Model(nn.Module):
    """
        CLD3 Model is based off the concept of Google's CLD3 model (https://github.com/google/cld3)
        General idea is that you extract some number of n-grams (by order).
        Each n-gram has an embedding associated with its hash.
        Each order is averaged across all n-grams from that order.
        All orders are concatenated together as input to the model (a simple MLP).
    """


    def __init__(self, vocab_size : int,
                        embedding_dim : int,
                        hidden_dim : int,
                        label_size : int,
                        num_ngram_orders : int,
                        montecarlo_layer : bool = False):
        super(CLD3Model, self).__init__()
        self.vocab_size = vocab_size
        self.num_ngram_orders = num_ngram_orders
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.label_size = label_size
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.hidden_layer = nn.Linear(embedding_dim*num_ngram_orders, hidden_dim)
        self.montecarlo_layer = montecarlo_layer
        if montecarlo_layer:
            self.proj = MCSoftmaxDenseFA(hidden_dim, label_size, 1, logits_only=True)
        else:
            self.proj = nn.Linear(hidden_dim, label_size)

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
        output = F.log_softmax(self.proj(hidden), dim=-1)
        return output

    def save_object(self):
        save = {
            'weights': self.state_dict(),
            'vocab_size': self.vocab_size,
            "num_ngram_orders": self.num_ngram_orders,
            "embedding_dim": self.embedding_dim,
            "hidden_dim": self.hidden_dim,
            "label_size": self.label_size,
            "montecarlo_layer": self.montecarlo_layer
        }
        return save

###################################### ROFORMER #######################################


class RoformerModel(nn.Module):
    """
        This model is based of the Huggingface Model implementation of the RoFormer Model
            Paper: https://arxiv.org/abs/2104.09864
            Huggingface: https://huggingface.co/docs/transformers/main/en/model_doc/roformer#transformers.RoFormerModel
    """

    def __init__(self,
                    vocab_size : int,
                    embedding_dim : int,
                    hidden_dim : int,
                    label_size : int,
                    num_layers : int,
                    max_len : int,
                    nhead : int = 8,
                    dropout : float = 0.1,
                    montecarlo_layer : bool = False
                    ):
        super(RoformerModel, self).__init__()
        self.config = RoFormerConfig(
            vocab_size = vocab_size,
            embedding_size = embedding_dim,
            hidden_size = hidden_dim,
            num_hidden_layers = num_layers,
            num_attention_heads = nhead,
            intermediate_size = hidden_dim,
            hidden_dropout_prob = dropout,
            attention_probs_dropout_prob = dropout,
            max_position_embeddings = max_len,
            type_vocab_size = 2,
            rotary_value = False
        )
        self.vocab_size = vocab_size
        self.embedding_size = embedding_dim
        self.hidden_size = hidden_dim
        self.num_hidden_layers = num_layers
        self.num_attention_heads = nhead
        self.intermediate_size = hidden_dim
        self.hidden_dropout_prob = dropout
        self.attention_probs_dropout_prob = dropout
        self.max_position_embeddings = max_len
        self.label_size = label_size
        self.montecarlo_layer = montecarlo_layer

        self.model = RoFormerModel(self.config)
        if montecarlo_layer:
            self.proj = MCSoftmaxDenseFA(hidden_dim, label_size, 1, logits_only=True)
        else:
            self.proj = nn.Linear(hidden_dim, label_size)

    def forward(self, inputs):
        inputs = inputs[:, :self.max_position_embeddings]
        mask = torch.tensor(inputs!=0, dtype=torch.float)
        encoding = torch.mean(self.model(inputs, attention_mask=mask).last_hidden_state, dim=1)
        output = self.proj(encoding)
        return F.log_softmax(output, dim=-1)

    def save_object(self):
        save = {
            "weights": self.state_dict(),
            "vocab_size" : self.vocab_size,
            "embedding_dim" : self.embedding_size,
            "hidden_dim" : self.hidden_size,
            "num_layers" : self.num_hidden_layers,
            "nhead" : self.num_attention_heads,
            "dropout": self.hidden_dropout_prob,
            "max_len": self.max_position_embeddings,
            "label_size": self.label_size,
            "montecarlo_layer": self.montecarlo_layer
        }
        return save

#################################### TRANSFORMER ######################################


class TransformerModel(nn.Module):
    """
        A basic transformer model that takes in input and averages across outputs as input to a linear model.
    """
    def __init__(self,
                    vocab_size : int,
                    embedding_dim : int,
                    hidden_dim : int,
                    label_size : int,
                    num_layers : int,
                    max_len : int,
                    nhead : int = 8,
                    roformer : bool = True,
                    montecarlo_layer : bool = False
                    ):
        super(TransformerModel, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(embedding_dim, nhead=nhead)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.embed = nn.Embedding(vocab_size, embedding_dim)
        self.pos_embed = PositionalEncoding(embed_size=embedding_dim, max_len=max_len)
        self.montecarlo_layer = montecarlo_layer
        if montecarlo_layer:
            self.proj = MCSoftmaxDenseFA(embedding_dim, label_size, 1, logits_only=True)
        else:
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
            "nhead": self.nhead,
            "montecarlo_layer": self.montecarlo_layer
        }
        return save

class PositionalEncoding(nn.Module):
    """
        Traditional sinosoidal positional encoding for transformer model.
    """
    
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

################################### CONVOLUTIONAL #####################################


class ConvModel(nn.Module):
    """
        Convolutional model which convolves over embedded tokens before passing to a linear layer.
    """

    def __init__(self,
                    vocab_size : int,
                    label_size : int,
                    embedding_dim : int = 256,
                    conv_min_width : int = 2,
                    conv_max_width : int = 5,
                    conv_depth : int = 64,
                    montecarlo_layer : bool = False):
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

        self.montecarlo_layer = montecarlo_layer
        if montecarlo_layer:
            self.proj = MCSoftmaxDenseFA(embedding_dim, label_size, 1, logits_only=True)
        else:
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
            "montecarlo_layer": self.montecarlo_layer
        }
        return save

####################################### UNET ##########################################

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
    """
        An implementation of the UNET model which convolves embeddings down before then upsampling them.
        This works at a sentence level and each token will have its own target label.
    """

    def __init__(self,
                    vocab_size : int,
                    label_size : int,
                    embedding_size : int = 128,
                    length : int = 1024):
        super(UNETModel, self).__init__()
        self.chs = [embedding_size*(2**i) for i in range(5)]

        self.vocab_size = vocab_size
        self.label_size = label_size
        self.embedding_dim = embedding_size
        self.length = length

        self.embed = nn.Embedding(vocab_size, embedding_size)
        self.encoder = Encoder(chs=self.chs)
        self.decoder = Decoder(chs=self.chs[::-1][:-1])
        self.proj = nn.Linear(self.chs[1], label_size)

    
    def forward(self, inputs):
        inputs = inputs[:, :self.length]
        emb = self.embed(inputs).transpose(1,2)
        encoding = self.encoder(emb)
        decoding = self.decoder(encoding[::-1][0], encoding[::-1][1:])
        upsampled = F.interpolate(decoding, self.length)
        out = self.proj(upsampled.transpose(1,2))

        return F.log_softmax(out, dim=2)

    def save_object(self):
        save = {
            "weights": self.state_dict(),
            "vocab_size": self.vocab_size,
            "embedding_size": self.embedding_dim,
            "label_size": self.label_size,
            "length": self.length,
            "montecarlo_layer": self.montecarlo_layer
        }
        return save


##################### FLASH MODEL #################

import copy
from typing import Optional, Any, Union, Callable, Tuple

import math
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

# Taken from facebookresearch/llama/model.py
def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


# Taken from facebookresearch/llama/model.py
def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)

    return xq_out.type_as(xq), xk_out.type_as(xk)

# Taken from facebookresearch/llama/model.py
def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (
        theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim)
    )
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64

    return freqs_cis


class TranformerLayer(nn.Module):
    """Transformer encoder block using F.scaled_dot_product_attention().

    This block has the following changes from a typical transformer encoder:

        - Rotary embeddings are applied to the key/query matrices.
        - Layer norm is applied before attention and feed forward, instead of
            after.
        - Keys arising from padding are masked during attention.
        - GELU activation is used instead of ReLU.

    Args:
        model_config (ModelConfig): Model config settings.
    """
    def __init__(self, d_model: int,
                        nhead: int,
                        dim_feedforward: int=2048,
                        dropout: float=0.1,
                        max_seq_len: int=256,
                        use_rotary: bool = False,
                        activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
                        layer_norm_eps: float=1e05, batch_first: bool = False, norm_first: bool = False,
                        device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__() 

        # self.pad_id = model_config.pad_id
        self.dropout = dropout
        self.nheads = nhead
        self.d_head = d_model // nhead
        # self.max_seq_len = model_config.max_seq_len

        # Attention
        self.q = nn.Linear(
            in_features=d_model,
            out_features=d_model,
            bias=False,
        )
        self.k = nn.Linear(
            in_features=d_model,
            out_features=d_model,
            bias=False,
        )
        self.v = nn.Linear(
            in_features=d_model,
            out_features=d_model,
            bias=False,
        )
        self.att_proj_linear = nn.Linear(
            in_features=d_model,
            out_features=d_model,
        )
        self.resid_dropout = nn.Dropout(dropout)

        # FF Layer
        self.ff_dropout = nn.Dropout(dropout)
        self.ff_linear_1 = nn.Linear(
            in_features=d_model,
            out_features=dim_feedforward,
        )
        self.ff_linear_2 = nn.Linear(
            in_features=dim_feedforward,
            out_features=d_model,
        )
        self.ff_activation = activation

        # Pre layer norms
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.max_seq_len = max_seq_len
        self.use_rotary = use_rotary
        if use_rotary:
            self.freqs_cis = precompute_freqs_cis(dim = d_model // nhead, end = self.max_seq_len * 2)


    def forward(
        self, 
        src: torch.Tensor, 
        src_mask: Optional[Tensor],
        src_key_padding_mask: torch.Tensor,
        is_causal: bool = False
    ):
        src_key_padding_mask = _canonical_mask(
            mask=src_key_padding_mask,
            mask_name="src_key_padding_mask",
            other_type=_none_or_dtype(src_mask),
            other_name="src_mask",
            target_type=src.dtype
        )

        src_mask = _canonical_mask(
            mask=src_mask,
            mask_name="src_mask",
            other_type=None,
            other_name="",
            target_type=src.dtype,
            check_other=False,
        )


        pad_mask, _ = self.merge_masks(src_mask, src_key_padding_mask, src)

        x = src
        x = x + self._att_block(self.norm1(x), pad_mask, is_causal=is_causal)
        x = x + self._ff_block(self.norm2(x))

        return x

    def _att_block(
        self, x: torch.Tensor, pad_mask: torch.Tensor, is_causal
    ):
        batch_size, seq_len, _ = x.shape
        xq, xk, xv = self.q(x), self.k(x), self.v(x)

        if self.use_rotary:
            # Reshape for rotary embeddings
            xq = xq.view(batch_size, seq_len, self.nheads, self.d_head)
            xk = xk.view(batch_size, seq_len, self.nheads, self.d_head)
            xv = xv.view(batch_size, seq_len, self.nheads, self.d_head)
            xq, xk = apply_rotary_emb(xq, xk, freqs_cis=self.freqs_cis[:seq_len])

            # Reshape for attention calculation: (b_sz, n_head, s_len, d_head)
            xq = xq.transpose(1, 2)
            xk = xk.transpose(1, 2)
            xv = xv.transpose(1, 2)

        # Required as we are not using a nn.Dropout layer
        if self.training:
            att_dropout = self.dropout
        else:
            att_dropout = 0.0

        # Using beta torch functionality (subject to change)
        # See - https://shorturl.at/jtI17
        att = F.scaled_dot_product_attention(
            query=xq,
            key=xk,
            value=xv,
            attn_mask=pad_mask,
            dropout_p=att_dropout,
            is_causal=is_causal,
        )

        # Shape (b_sz, s_len, n_head, d_head)
        out = att.transpose(1, 2).contiguous()
        out = out.view(batch_size, seq_len, self.nheads * self.d_head)

        return self.resid_dropout(self.att_proj_linear(out))

    def _ff_block(self, x: torch.Tensor):
        x = self.ff_linear_2(self.ff_activation(self.ff_linear_1(x)))

        return self.ff_dropout(x)

    def merge_masks(self, attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor],
                    query: Tensor) -> Tuple[Optional[Tensor], Optional[int]]:
        r"""
        Determine mask type and combine masks if necessary. If only one mask is provided, that mask
        and the corresponding mask type will be returned. If both masks are provided, they will be both
        expanded to shape ``(batch_size, num_heads, seq_len, seq_len)``, combined with logical ``or``
        and mask type 2 will be returned
        Args:
            attn_mask: attention mask of shape ``(seq_len, seq_len)``, mask type 0
            key_padding_mask: padding mask of shape ``(batch_size, seq_len)``, mask type 1
            query: query embeddings of shape ``(batch_size, seq_len, embed_dim)``
        Returns:
            merged_mask: merged mask
            mask_type: merged mask type (0, 1, or 2)
        """
        mask_type: Optional[int] = None
        merged_mask: Optional[Tensor] = None

        if attn_mask is not None:
            mask_type = 0
            merged_mask = attn_mask
        if key_padding_mask is not None:
            mask_type = 1
            merged_mask = key_padding_mask
        if (attn_mask is not None) and (key_padding_mask is not None):
            # In this branch query can't be a nested tensor, so it has a shape
            batch_size, seq_len, _ = query.shape
            mask_type = 2
            key_padding_mask_expanded = key_padding_mask.view(batch_size, 1, 1, seq_len) \
                                                        .expand(-1, self.num_heads, -1, -1)
            attn_mask_expanded = attn_mask.view(1, 1, seq_len, seq_len).expand(batch_size, self.num_heads, -1, -1)
            merged_mask = attn_mask_expanded + key_padding_mask_expanded
        return merged_mask, mask_type


class FlashModel(nn.Module):
    def __init__(self,
                    vocab_size : int,
                    embedding_dim : int,
                    hidden_dim : int,
                    label_size : int,
                    num_layers : int,
                    max_len : int,
                    nhead : int = 8,
                    dropout : float = 0.1,
                    montecarlo_layer : bool = False,
                    use_rotary: bool = False,
                    ):
        super(FlashModel, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.label_size = label_size
        self.num_layers = num_layers
        self.max_length = max_len
        self.nhead = nhead
        self.dropout = dropout
        self.montecarlo_layer = montecarlo_layer
        self.use_rotary = use_rotary

        layer = TranformerLayer(d_model=embedding_dim,
                                nhead=nhead,
                                dropout=dropout,
                                max_seq_len=max_len,
                                use_rotary=use_rotary,
                                )

        self.encoder = nn.TransformerEncoder(layer, num_layers=6)
        self.embed = nn.Embedding(vocab_size, embedding_dim)
        self.pos_embed = PositionalEncoding(embed_size=embedding_dim, max_len=max_len)
        self.montecarlo_layer = montecarlo_layer
        if montecarlo_layer:
            self.proj = MCSoftmaxDenseFA(embedding_dim, label_size, 1, logits_only=True)
        else:
            self.proj = nn.Linear(embedding_dim, label_size)

    def forward(self, inputs):
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
            "nhead": self.nhead,
            "montecarlo_layer": self.montecarlo_layer,
            "use_rotary": self.use_rotary,
        }
        return save


def _canonical_mask(
        mask: Optional[Tensor],
        mask_name: str,
        other_type: Optional[DType],
        other_name: str,
        target_type: DType,
        check_other: bool = True,
) -> Optional[Tensor]:

    if mask is not None:
        _mask_dtype = mask.dtype
        _mask_is_float = torch.is_floating_point(mask)
        if _mask_dtype != torch.bool and not _mask_is_float:
            raise AssertionError(
                f"only bool and floating types of {mask_name} are supported")
        if check_other and other_type is not None:
            if _mask_dtype != other_type:
                warnings.warn(
                    f"Support for mismatched {mask_name} and {other_name} "
                    "is deprecated. Use same type for both instead."
                )
        if not _mask_is_float:
            mask = (
                torch.zeros_like(mask, dtype=target_type)
                .masked_fill_(mask, float("-inf"))
            )
    return mask

def _none_or_dtype(input: Optional[Tensor]) -> Optional[DType]:
    if input is None:
        return None
    elif isinstance(input, torch.Tensor):
        return input.dtype
    raise RuntimeError("input to _none_or_dtype() must be None or torch.Tensor")