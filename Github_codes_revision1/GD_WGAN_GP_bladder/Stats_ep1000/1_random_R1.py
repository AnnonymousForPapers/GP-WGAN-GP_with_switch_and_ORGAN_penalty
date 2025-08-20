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
Program to run random generator to generate Pseudo-immunogenic sequences for HLA-A*0201
python 3.9.15
'''

import random
import pandas as pd
import os


# The alphabet of characters
chars = "ARNDCQEGHILKMFPSTWYV"

def generate_sequences(n_sequences=100):
    sequences = []
    for _ in range(n_sequences):
        # Step 1: sample sequence length (either 9 or 10)
        seq_len = random.choice([9, 10])
        # Step 2: generate sequence
        seq = "".join(random.choice(chars) for _ in range(seq_len))
        sequences.append(seq)
    return sequences

method = "random_generator"
num_epochs = 1000 # generator trained epoch
epoch_file = 1000

#batch= args.batch
batch_size = 10000 # 6232 peptides = len(raw_Bladder)

#%% Set file directory
data_file = '../../'
outdir = data_file + 'result/' + method + '/epoch' + str(epoch_file)
print("outdir is {}".format(outdir))
#outname = 'deepimmuno-GANRL-bladder-epoch' + str(num_epochs) + '.txt'
outname = 'deepimmuno-GANRL-bladder-epoch' + str(num_epochs) + '-batch' + str(batch_size) + '.txt'

#%%
pseudo = generate_sequences(batch_size)
df = pd.DataFrame({'peptide': pseudo, 'HLA': ['HLA-A*0201' for i in range(len(pseudo))],
                    'immunogenicity': [1 for i in range(len(pseudo))]})
df.to_csv(os.path.join(outdir,outname),sep='\t',index=None)

