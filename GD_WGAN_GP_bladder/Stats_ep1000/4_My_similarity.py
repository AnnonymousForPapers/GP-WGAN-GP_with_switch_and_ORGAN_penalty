# -*- coding: utf-8 -*-
'''
Revised from https://github.com/frankligy/DeepImmuno
DeepImmuno: deep learning-empowered prediction and generation of 
immunogenic peptides for T-cell immunity, Briefings in Bioinformatics, 
May 03 2021 (https://doi.org/10.1093/bib/bbab160)
'''
"""
Created on Thu Mar 30 13:56:41 2023

@author: ych22001
"""

# sequence matching
import pandas as pd
import numpy as np

#%% Set file directory
data_file = '../../'

#%%
#data = pd.read_csv('D:/PhD thesis/GCN/My Code/data/gan_a0201.csv')
data = pd.read_csv(data_file + 'data/neoepitopes/Bladder.4.0_test_mut.csv')
raw = data['peptide'].values
from difflib import SequenceMatcher
def similar(a,b):
    return SequenceMatcher(None,a,b).ratio()

"""
Test peptides data
"""
# method = 'GAN_RL_old'
# method = 'GAN_RL_imm_select_LG_mean_only'
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
epoch = 1000
num_epochs = 1000
batch_size = 10000 # 6232 peptides = len(raw_Bladder)
ep = 'epoch' + str(epoch)
# data_path = 'D:/PhD thesis/GCN/My Code/' + method + '/' + ep + '/deepimmuno-GANRL-bladder-' + ep + '_rmv.txt'
for method in method_array:
    data_path = data_file + 'result/' + method + '/' + ep + '/deepimmuno-GANRL-bladder-epoch' + str(num_epochs) + '-batch' + str(batch_size) + '_rmv.txt'
    generate = pd.read_csv(data_path,sep='\t')
    
    #generate = pd.read_csv('D:/PhD thesis/GCN/My Code/data/df/df_all_epoch100.csv')
    #generate = pd.read_csv('D:/PhD thesis/GCN/My Code/Test/GAN_RL_no_label_result/epoch0/test_generator_0.csv')
    seq = generate['peptide'].values
    
    len_seq = len(seq)
    count = 0
    whole1 = []  # store mean value
    whole2 = []  # store max value
    for item1 in seq:
        total = []
        for item2 in raw:
            total.append((item2, similar(item1, item2)))
        total = np.array(sorted(total, reverse=True, key=lambda x: x[1]))[:,1].astype(np.float64)
        count = count + 1
        if not (count%100):
            print(str(count)+'/'+str(len_seq))
        whole1.append(total.mean())
        whole2.append(total.max())
    
    # import matplotlib as mpl
    # import matplotlib.pyplot as plt
    # mpl.rcParams['pdf.fonttype'] = 42
    # mpl.rcParams['ps.fonttype'] = 42
    # mpl.rcParams['font.family'] = 'Arial'
    
    # import seaborn as sns
    # #increase font size of all elements
    # sns.reset_defaults()
    # #sns.set(font_scale=1.0)
    # plot = sns.displot(data=pd.DataFrame({'Max Similarity': whole2}),x='Max Similarity',kde=True, aspect=1.5)
    # plt.grid(b=True,axis='y',alpha=0.3)
    # #plt.xticks(np.arange(0.3, 0.05, 1))
    # # plot.fig.set_figwidth(5)
    # # plot.fig.set_figheight(10)
    # plot.set_xticklabels(size = 15)
    # plot.set_yticklabels(size = 15)
    # plt.title('Epoch '+ str(num_epochs), fontsize=20)
    # plt.xlabel('Max Similarity',fontsize=20);
    # plt.ylabel('Count',fontsize=20);
    # plt.show()
    
    # plt.rcParams.update({'font.size': 15}) # must set in top
    # df=pd.DataFrame({'Max Similarity': whole2})
    # num = df['Max Similarity'].unique()
    # hist = df.hist(bins=len(num), color='#86bf91', edgecolor = 'black' , rwidth=0.5, figsize=(8,6))  
    # plt.grid(b=True,axis='y',alpha=0.3)
    # plt.title('Epoch '+ str(epoch), fontsize=20)
    # plt.xlabel('Max Similarity',fontsize=20)
    # plt.ylabel('Count',fontsize=20)
    # plt.show()
    
    outdir = data_file + 'result/' + method + '/' + ep + '/'
    print("outdir is {}".format(outdir))
    # plot.savefig(outdir + '/similarity_epoch' + str(num_epochs) + '.pdf')
    # np.sum(np.where(np.array(whole2) > 0.75, 1,0))  # 891- 60%
    
    #np.save(outdir + '/MaxSimilarity_' + ep  + '-batch' + str(batch_size), whole2)
    np.save(outdir + '/MaxSimilarity_epoch' + str(num_epochs) + '-batch' + str(batch_size), whole2)
    
    count = 0
    #with open('D:/PhD thesis/GCN/My Code/' + method + '/' + ep + '/log_rmv_0.txt','w') as f:
    with open(data_file + 'result/' + method + '/' + ep + '/epoch' + str(num_epochs) + '-batch' + str(batch_size) +'log_rmv_0.txt','w') as f:
        for item1 in seq:
            total = []
            for item2 in raw:
                total.append((item2,similar(item1,item2)))
            total = sorted(total,reverse=True,key=lambda x:x[1])[:5]
            print('{0}:{1}'.format(item1,total),file=f)
            count = count + 1
            if not (count%10):
                print(str(count)+'/'+str(len_seq))
    
    print(f'\n Method: {method}')
    print(f"Number of same peptides: {whole2.count(1.0)}")