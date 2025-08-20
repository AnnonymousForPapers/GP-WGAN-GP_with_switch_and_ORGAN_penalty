# -*- coding: utf-8 -*-
"""
Created on Thu Mar 30 09:51:30 2023

@author: ych22001
"""

import pandas as pd

#%% Set file directory
data_file = '../../'

#%%
# method = 'GD_WGAN_wclip_MolGAN_Loss_MyPredictor'
# method = 'GAN_RL_imm_select_LG_mean_only'
method_array = ["GD_WGAN_wclip_MolGAN_DScalar_Loss_MyPredictor",
                "GD_WGAN_wclip_MolGAN_Gamma0_DScalar_Loss_MyPredictor",
                "GD_WGAN_wclip_OR_MolGAN_Gamma0_DScalar_Loss_MyPredictor"]
ep = 'epoch1000'

# num_epochs = 1000 # generator trained epoch
num_epochs_array = [816,6,6] # generator trained epoch
batch_size = 10000 # number of generated peptides
for method, num_epochs  in zip(method_array, num_epochs_array):
    # Database: http://biopharm.zju.edu.cn/tsnadb/
    #Bladder = pd.read_csv('D:/PhD thesis/GCN/My Code/' + method + '/' + ep + '/deepimmuno-GANRL-bladder-' + ep + '.txt',sep='\t')
    Bladder = pd.read_csv(data_file + 'result/' + method + '/' + ep + '/best_deepimmuno-GANRL-bladder-epoch' + str(num_epochs) + '-batch' + str(batch_size) + '.txt',sep='\t')
    # https://pandas.pydata.org/docs/reference/api/pandas.Series.str.count.html
    Bladder_count = Bladder['peptide'].str.count('-')
    # https://stackoverflow.com/questions/17071871/how-do-i-select-rows-from-a-dataframe-based-on-column-values
    Bladder_rmv = Bladder.loc[Bladder_count<2]
    Bladder_out = Bladder_rmv
    Bladder_out['peptide'] = Bladder_rmv['peptide'].str.replace('-', '')
    
    unique_rmv_peptides_df = Bladder_out.drop_duplicates(subset='peptide').reset_index(drop=True)
    num_unique_rmv_peptides = len(unique_rmv_peptides_df)
    print(f'Method: {method}')
    print(f"Number of peptide sequences with 9 and 10 mers: {len(Bladder_out['peptide'])}")
    print(f'Number of unique peptide sequences: {num_unique_rmv_peptides}')
    unique_peptides_df = Bladder.drop_duplicates(subset='peptide').reset_index(drop=True)
    num_unique_peptides = len(unique_peptides_df)
    print(f'Number of unique peptide sequences before removing: {num_unique_peptides}')
    
    import os
    outdir = data_file + 'result/' + method + '/' + ep + '/'
    print("outdir is {}".format(outdir))
    outname = 'best_deepimmuno-GANRL-bladder-epoch' + str(num_epochs) + '-batch' + str(batch_size) + '_rmv.txt'
    # Bladder_out.to_csv(os.path.join(outdir,outname),sep='\t',index=None)
    unique_rmv_peptides_df.to_csv(os.path.join(outdir,outname),sep='\t',index=None)
    
    # https://pandas.pydata.org/docs/reference/api/pandas.Series.str.len.html
    #Bladder_length = Bladder['peptide'].str.len()
    # https://stackoverflow.com/questions/17071871/how-do-i-select-rows-from-a-dataframe-based-on-column-values
    #Bladder_out = Bladder.loc[Bladder['peptide'].str.len()>=9]
    


