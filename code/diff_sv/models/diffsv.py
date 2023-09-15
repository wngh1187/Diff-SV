import torch
import torch.nn as nn
import numpy as np
from models.modules import SEBasicBlock
from models.denoiser import Denoiser as Denoiser
from models.transformer import Transformer
import torch.nn.functional as F
import torchaudio

from models.base import BaseModule
import math

class PreEmphasis(torch.nn.Module):

    def __init__(self, coef: float = 0.97):
        super().__init__()
        self.coef = coef
        self.register_buffer(
            'flipped_filter', torch.FloatTensor([-self.coef, 1.]).unsqueeze(0).unsqueeze(0)
        )

    def forward(self, input: torch.tensor) -> torch.tensor:
        input = input.unsqueeze(1)
        input = F.pad(input, (1, 0), 'reflect')
        return F.conv1d(input, self.flipped_filter).squeeze(1)


class DiffSV(BaseModule): 
    def __init__(self, args):
        super(DiffSV, self).__init__()
        self.max_epoch = args['epoch']
        self.l_channel = args['l_channel']
        self.l_num_convblocks = args['l_num_convblocks']
        self.code_dim = args['code_dim']
        self.stride = args['stride']
        self.first_kernel_size = args['first_kernel_size']
        self.first_stride_size = args['first_stride_size']
        self.first_padding_size = args['first_padding_size']

        self.torchfbank = nn.Sequential(
            PreEmphasis(),            
            torchaudio.transforms.MelSpectrogram(sample_rate=args['samplerate'], n_fft=args['nfft'], win_length=args['winlen'], hop_length=args['winstep'], \
                                                 f_min = args['f_min'], f_max = args['f_max'], window_fn=args['winfunc'], n_mels=args['nfilts']),
            )
        
        self.inplanes   = self.l_channel[0]
        
        ### define enhancer ###
        self.enhancer = Transformer(4, args['nfilts'], d_inner=512)

        #### define denoiser ###
        self.denoiser = Denoiser(args['nfilts'])

        ### define extractor ###
        self.conv1 = nn.Conv2d(3, self.l_channel[0] , kernel_size=self.first_kernel_size, stride=self.first_stride_size, padding=self.first_padding_size,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(self.l_channel[0])
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(SEBasicBlock, self.l_channel[0], self.l_num_convblocks[0], stride=self.stride[0])
        self.layer2 = self._make_layer(SEBasicBlock, self.l_channel[1], self.l_num_convblocks[1], stride=self.stride[1])
        self.layer3 = self._make_layer(SEBasicBlock, self.l_channel[2], self.l_num_convblocks[2], stride=self.stride[2])
        self.layer4 = self._make_layer(SEBasicBlock, self.l_channel[3], self.l_num_convblocks[3], stride=self.stride[3])

        final_dim = self.l_channel[-1] * args['nfilts']//8
        
        self.attention = nn.Sequential(
            nn.Conv1d(final_dim, final_dim//8, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(final_dim//8),
            nn.Conv1d(final_dim//8, final_dim, kernel_size=1), 
            nn.Softmax(dim=-1),
        )
        
        self.bn_agg = nn.BatchNorm1d(final_dim * 2)

        self.fc = nn.Linear(final_dim*2, self.code_dim)
        self.bn_code = nn.BatchNorm1d(self.code_dim)
        
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

        
    def forward(self, x, ref_x=None, aug=None, epoch=None, is_train=True):
        if is_train:
            with torch.no_grad():
                x = self.torchfbank(x)+1e-6
                x = x.log()   
                x = x - torch.mean(x, dim=-1, keepdim=True)
                org_x = x

                ref_x = self.torchfbank(ref_x)+1e-6
                ref_x = ref_x.log()   
                ref_x = ref_x - torch.mean(ref_x, dim=-1, keepdim=True)

            x_hat = self.enhancer(org_x)
            prior_loss = F.mse_loss(ref_x, x_hat)

            z0_hat, ddim_loss = self.denoiser(x_hat, ref_x)

            z0_hat = z0_hat.detach()
        else:
            with torch.no_grad():
                x = self.torchfbank(x)+1e-6
                x = x.log()   
                x = x - torch.mean(x, dim=-1, keepdim=True)
                org_x = x

            x_hat = self.enhancer(org_x)
            z0_hat = self.denoiser(x_hat, is_inference=True)

        x = torch.stack([org_x, x_hat, z0_hat], 1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        bs, _, _, time = x.size()
        x = x.reshape(bs, -1, time)
        
        w = self.attention(x)
        m = torch.sum(x * w, dim=-1)
        s = torch.sqrt((torch.sum((x ** 2) * w, dim=-1) - m ** 2).clamp(min=1e-5))
        x = torch.cat([m, s], dim=1)
        x =    self.bn_agg(x)

        code = self.fc(x)
        code = self.bn_code(code)

        
        if is_train:
            return code, prior_loss, ddim_loss
        else:
            return code