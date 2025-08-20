# -*- coding: utf-8 -*-
'''
Revised from https://github.com/frankligy/DeepImmuno
DeepImmuno: deep learning-empowered prediction and generation of 
immunogenic peptides for T-cell immunity, Briefings in Bioinformatics, 
May 03 2021 (https://doi.org/10.1093/bib/bbab160)
'''
"""
Created on Wed Feb 15 08:29:11 2023

@author: ych22001
"""

'''
Program to run deepimmuno-GAN to generate Pseudo-immunogenic sequences for HLA-A*0201
python 3.9.15
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import os
import random

random.seed(53)
random.random()

torch.manual_seed(53)
torch.rand(4)

np.random.seed(53)
np.random.rand(4)

# build the model
class ResBlock(nn.Module):
    def __init__(self, hidden):  # hidden means the number of filters
        super(ResBlock, self).__init__()
        self.res_block = nn.Sequential(
            nn.ReLU(True),  # in_place = True
            # https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
            # input shape: hidden, output shape: hidden
            # https://discuss.pytorch.org/t/input-form-of-conv1d/153775
            # nn.Conv1d expects a 3-dimensional input in the shape [batch_size, channels, seq_len]
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=1), # padding: output channel has the same size as the input channel
            nn.ReLU(True),
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=1),
        )

    def forward(self, input):  # input [N, hidden, seq_len]
        output = self.res_block(input)
        return input + 0.3 * output  # [N, hidden, seq_len]  doesn't change anything

#%% CNN Generator
class Generator(nn.Module):
    def __init__(self,hidden,seq_len,n_chars,batch_size):
        super(Generator,self).__init__()
        self.fc1 = nn.Linear(128,hidden*seq_len)
        self.block = nn.Sequential(
            ResBlock(hidden),
            ResBlock(hidden),
            ResBlock(hidden),
            ResBlock(hidden),
            ResBlock(hidden),
        )
        self.conv1 = nn.Conv1d(hidden,n_chars,kernel_size=1)
        self.hidden = hidden
        self.seq_len = seq_len
        self.n_chars = n_chars
        self.batch_size = batch_size

    def forward(self,noise):  # noise [batch,128]
        output = self.fc1(noise)    # [batch,hidden*seq_len]
        output = output.view(-1,self.hidden,self.seq_len)   # [batch,hidden,seq_len]
        output = self.block(output)  # [batch,hidden,seq_len]
        output = self.conv1(output)  # [batch,n_chars,seq_len]
        '''
        In order to understand the following step, you have to understand how torch.view actually work, it basically
        alloacte all entry into the resultant tensor of shape you specified. line by line, then layer by layer.
        
        Also, contiguous is to make sure the memory is contiguous after transpose, make sure it will be the same as 
        being created form stracth
        '''
        output = output.transpose(1,2)  # [batch,seq_len,n_chars]
        output = output.contiguous()
        output = output.view(self.batch_size*self.seq_len,self.n_chars)
        output = F.gumbel_softmax(output,tau=0.75,hard=False)  # github code tau=0.5, paper tau=0.75  [batch*seq_len,n_chars]
        output = output.view(self.batch_size,self.seq_len,self.n_chars)   # [batch,seq_len,n_chars]
        return output
    
#%% LSTM Generator
class GeneratorLSTM(nn.Module):
    def __init__(self,hidden,seq_len,n_chars,batch_size, num_layers=3, bidirectional=False):
        super(GeneratorLSTM,self).__init__()
        self.fc1 = nn.Linear(128,hidden*seq_len)
        # LSTM over the sequence (replaces ResBlock stack + conv1)
        self.lstm = nn.LSTM(
            input_size=hidden,
            hidden_size=hidden,
            num_layers=num_layers,
            batch_first=True,        # inputs/outputs are [B, T, C]
            bidirectional=bidirectional
        )
        # Per-timestep classifier to n_chars (like your 1x1 conv)
        out_dim = hidden * (2 if bidirectional else 1)
        self.proj = nn.Linear(out_dim, n_chars)
        self.hidden = hidden
        self.seq_len = seq_len
        self.n_chars = n_chars
        self.batch_size = batch_size

    def forward(self,noise):  # noise [batch,128]
        output = self.fc1(noise)    # [batch,hidden*seq_len]
        # output = output.view(-1,self.hidden,self.seq_len)   # [batch,hidden,seq_len]
        output = output.view(-1, self.seq_len, self.hidden)            # [B, T, C]
        
        
        # output = self.block(output)  # [batch,hidden,seq_len]
        # 2) sequence modeling with LSTM
        #    (h0, c0 default to zeros; you could also learn them if you like)
        output, _ = self.lstm(output)                                   # [B, T, C or 2C]
        
        # output = self.conv1(output)  # [batch,n_chars,seq_len]
        # output = output.transpose(1,2)  # [batch,seq_len,n_chars]
        # 3) per-timestep logits -> [B, T, n_chars]
        output = self.proj(output)                                      # [B, T, n_chars]

        '''
        In order to understand the following step, you have to understand how torch.view actually work, it basically
        alloacte all entry into the resultant tensor of shape you specified. line by line, then layer by layer.
        
        Also, contiguous is to make sure the memory is contiguous after transpose, make sure it will be the same as 
        being created form stracth
        '''
        output = output.contiguous()
        output = output.view(self.batch_size*self.seq_len,self.n_chars)
        output = F.gumbel_softmax(output,tau=0.75,hard=False)  # github code tau=0.5, paper tau=0.75  [batch*seq_len,n_chars]
        output = output.view(self.batch_size,self.seq_len,self.n_chars)   # [batch,seq_len,n_chars]
        return output
    
#%% Transformer Generator
def build_rotary_embeddings(seq_len, head_dim, base=10000):
    pos = torch.arange(seq_len, dtype=torch.float32) # [0, 1, ..., seq_len]
    freqs = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
    # freqs = (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
    # torch.arange(0, head_dim, 2).float() -> [0, 2, 4, ..., head_dim]
    angles = torch.outer(pos, freqs)
    return torch.cos(angles), torch.sin(angles)

def apply_rotary(x, cos, sin):
    # print(f"x.shape: {x.shape}")
    # print(f"cos.shape: {cos.shape}")
    # print(f"sin.shape: {sin.shape}")
    x1, x2 = x[..., ::2], x[..., 1::2] # even position, odd position of x
    x_rot = torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    return x_rot.flatten(-2)

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        norm = x.norm(dim=-1, keepdim=True)
        return self.weight * x / (norm / (x.size(-1) ** 0.5) + self.eps)

class GQAAttention(nn.Module):
    def __init__(self, embed_dim, q_heads, kv_heads):
        super().__init__()
        assert embed_dim % q_heads == 0
        self.q_heads = q_heads
        self.kv_heads = kv_heads
        self.head_dim = embed_dim // q_heads
        self.scale = self.head_dim ** 0.5

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, self.head_dim * kv_heads)
        self.v_proj = nn.Linear(embed_dim, self.head_dim * kv_heads)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x, cos, sin):
        B, T, _ = x.size()
        q = self.q_proj(x).view(B, T, self.q_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.kv_heads, self.head_dim).transpose(1, 2)

        k = k.repeat_interleave(self.q_heads // self.kv_heads, dim=1)
        v = v.repeat_interleave(self.q_heads // self.kv_heads, dim=1)

        q = apply_rotary(q, cos, sin)
        k = apply_rotary(k, cos, sin)

        att = (q @ k.transpose(-2, -1)) / self.scale
        # att = att.masked_fill(torch.triu(torch.ones(T, T, device=x.device), 1) == 1, float('-inf'))
        weights = F.softmax(att, dim=-1)
        out = weights @ v
        return self.out_proj(out.transpose(1, 2).reshape(B, T, -1))

class FFNLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(dim, int(4*dim)), # 2048/4=512=4*emb_dim
            nn.GELU(),
            nn.Linear(int(4*dim), dim)
        )

    def forward(self, x):
        return self.ffn(x)
    
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, q_heads, kv_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.q_heads = q_heads
        self.attn = GQAAttention(embed_dim, q_heads, kv_heads)
        self.rms1 = RMSNorm(embed_dim)
        self.ffn = FFNLayer(embed_dim)
        self.rms2 = RMSNorm(embed_dim)

    def forward(self, x):
        seq_len = x.size(1)
        head_dim = self.embed_dim // self.q_heads
        cos, sin = build_rotary_embeddings(seq_len, head_dim)
        
        # Move to same device as x
        device = x.device
        cos = cos.to(device).unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, head_dim]
        sin = sin.to(device).unsqueeze(0).unsqueeze(0)

        x = self.attn(x, cos, sin) + x
        x = self.rms1(x)
        x = self.ffn(x) + x
        x = self.rms2(x)
        return x

class GeneratorTransformer(nn.Module):
    def __init__(self,hidden,seq_len,n_chars,batch_size, num_layers, q_heads, kv_heads):
        super(GeneratorTransformer,self).__init__()
        self.fc1 = nn.Linear(128,hidden*seq_len)
        # LSTM over the sequence (replaces ResBlock stack + conv1)
        self.layers = nn.ModuleList([
            TransformerBlock(hidden, q_heads, kv_heads) for _ in range(num_layers)
        ])
        self.proj = nn.Linear(hidden, n_chars)
        self.hidden = hidden
        self.seq_len = seq_len
        self.n_chars = n_chars
        self.batch_size = batch_size

    def forward(self,noise):  # noise [batch,128]
        output = self.fc1(noise)    # [batch,hidden*seq_len]
        # output = output.view(-1,self.hidden,self.seq_len)   # [batch,hidden,seq_len]
        output = output.view(-1, self.seq_len, self.hidden)            # [B, T, C]
        
        
        # output = self.block(output)  # [batch,hidden,seq_len]
        for layer in self.layers:
            output = layer(output)                                    # [B, T, C or 2C]
        
        # output = self.conv1(output)  # [batch,n_chars,seq_len]
        # output = output.transpose(1,2)  # [batch,seq_len,n_chars]
        # 3) per-timestep logits -> [B, T, n_chars]
        output = self.proj(output)                                      # [B, T, n_chars]

        '''
        In order to understand the following step, you have to understand how torch.view actually work, it basically
        alloacte all entry into the resultant tensor of shape you specified. line by line, then layer by layer.
        
        Also, contiguous is to make sure the memory is contiguous after transpose, make sure it will be the same as 
        being created form stracth
        '''
        output = output.contiguous()
        output = output.view(self.batch_size*self.seq_len,self.n_chars)
        output = F.gumbel_softmax(output,tau=0.75,hard=False)  # github code tau=0.5, paper tau=0.75  [batch*seq_len,n_chars]
        output = output.view(self.batch_size,self.seq_len,self.n_chars)   # [batch,seq_len,n_chars]
        return output

#%% Start     

# post utils functions
def inverse_transform(hard):  # [N,seq_len]
    amino = 'ARNDCQEGHILKMFPSTWYV-'
    result = []
    for row in hard:
        temp = ''
        for col in row:
            aa = amino[col]
            temp += aa
        result.append(temp)
    return result

method_array = ["Goal-directed_WGAN-GP_LSTM",
                "Goal-directed_WGAN-GP_NoSwithORGAN",
                "Goal-directed_WGAN-GP_ORGAN_LSTM",
                "Goal-directed_WGAN-GP_ORGAN_TransformerNoMaskL2",
                "Goal-directed_WGAN-GP_TransformerNoMaskL2"]

#batch= args.batch
batch_size = 10000 # 6232 peptides = len(raw_Bladder)
epoch_file = 1000
num_epochs = 1000
device = 'cpu'

noise = torch.randn(batch_size,128).to(device)  # [N, 128]

for method in method_array:

    #%% Set file directory
    data_file = '../../'
    outdir = data_file + 'result/' + method + '/epoch' + str(epoch_file)
    print("outdir is {}".format(outdir))
    #outname = 'deepimmuno-GANRL-bladder-epoch' + str(num_epochs) + '.txt'
    outname = 'deepimmuno-GANRL-bladder-epoch' + str(num_epochs) + '-batch' + str(batch_size) + '.txt'
    
    # generation = sample_generator(64).detach().cpu().numpy() # [N,seq_len,n_chars] # [64,10,21] # [?, peptide length, amino acids+'-']
    # auxiliary function during training GAN
    
    #%%
    seq_len = 10
    hidden = 128
    n_chars = 21
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # # https://stackoverflow.com/questions/50954479/using-cuda-with-pytorch
    # # tensor.to(device) command to move a whole model to a device
    if 'LSTM' in method:
        n_layers = 4
        is_bidirectional = False
        G = GeneratorLSTM(hidden, seq_len, n_chars, batch_size, n_layers, is_bidirectional).to(device)
    elif 'Transformer' in method:
        n_layers = 2
        q_heads = 8
        kv_heads = 8
        G = GeneratorTransformer(hidden, seq_len, n_chars, batch_size, n_layers, q_heads, kv_heads).to(device)
    else:
        G = Generator(hidden,seq_len,n_chars,batch_size).to(device)
    # #G.load_state_dict(torch.load('./models/wassGAN_G.pth'))
    G.load_state_dict(torch.load(data_file + 'result/' + method + '/epoch' + str(epoch_file) + '/model_epoch_' + str(num_epochs) + '.pth'))
    
    
    generated_data = G(noise)
    
    #noise = torch.randn(batch_size, 128).to(device)  # [N, 128]
    #generated_data = G(noise)  # [N, seq_len, n_chars]
    
    generation = generated_data.detach().cpu().numpy()
    # # https://numpy.org/doc/stable/reference/generated/numpy.argmax.html
    # # Returns the indices of the maximum values along an axis
    hard = np.argmax(generation, axis=2)  # [N,seq_len]
    pseudo = inverse_transform(hard)
    df = pd.DataFrame({'peptide': pseudo, 'HLA': ['HLA-A*0201' for i in range(len(pseudo))],
                        'immunogenicity': [1 for i in range(len(pseudo))]})
    df.to_csv(os.path.join(outdir,outname),sep='\t',index=None)

