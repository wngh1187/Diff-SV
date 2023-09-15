import torch
import torch.nn as nn
import numpy as np
from models.modules import Mish, MultiHeadAttention, PositionwiseFeedForward
from models.base import BaseModule

class Transformer(BaseModule):
    """ Transformer """
    def __init__(self, n_layers, d_model, n_head=2, d_inner=128):
        super(Transformer, self).__init__()
        self.n_layers = n_layers
        self.d_model = d_model
        self.n_head = n_head
        self.d_k = self.d_model // self.n_head
        self.d_v = self.d_model // self.n_head
        self.d_inner = d_inner
        self.fft_conv1d_kernel_size = [3, 3]
        self.dropout = 0.1
        
        self.spectral = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            Mish(),
            nn.Dropout(0.1),
            nn.Linear(self.d_model, self.d_model),
            Mish(),
            nn.Dropout(0.1)
        )

        self.layer_stack = nn.ModuleList([FFTBlock(
            self.d_model, self.d_inner, self.n_head, self.d_k, self.d_v, 
            self.fft_conv1d_kernel_size, self.dropout) for _ in range(self.n_layers)])

    def forward(self, x):
        x = self.spectral(x.transpose(1, 2))

        for enc_layer in self.layer_stack:
            x = enc_layer(x)
            
        return x.transpose(1, 2)

class FFTBlock(BaseModule):
    ''' FFT Block '''
    def __init__(self, d_model, d_inner, n_head, d_k, d_v, fft_conv1d_kernel_size, dropout):
        super(FFTBlock, self).__init__()
        self.slf_attn = MultiHeadAttention(
            n_head, d_model, d_k, d_v, dropout=dropout, spectral_norm=False)

        self.pos_ffn = PositionwiseFeedForward(
            d_model, d_inner, fft_conv1d_kernel_size, dropout=dropout)

    def forward(self, input):
        # multi-head self attn
        output = self.slf_attn(input)

        # position wise FF
        output = self.pos_ffn(output)

        return output


