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

method_array = ["Goal-directed_WGAN-GP",
                "WGAN-GP",
                "Goal-directed_WGAN-GP_Gamma0_25",
                "Goal-directed_WGAN-GP_Gamma0_5",
                "Goal-directed_WGAN-GP_Gamma0_75",
                "Goal-directed_WGAN-GP_noRewardNet",
                "Goal-directed_WGAN-GP_noSscale",
                "GD_WGAN_wclip_MolGAN_Gamma0_DScalar_Loss_MyPredictor",
                "GD_WGAN_wclip_MolGAN_DScalar_Loss_MyPredictor",
                "GD_WGAN_wclip_OR_MolGAN_Gamma0_DScalar_Loss_MyPredictor",
                "Goal-directed_WGAN-GP_ORGAN"]

#batch= args.batch
batch_size = 10000 # 6232 peptides = len(raw_Bladder)
method = 'GD_WGAN_wclip_MolGAN_Loss_MyPredictor'
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

