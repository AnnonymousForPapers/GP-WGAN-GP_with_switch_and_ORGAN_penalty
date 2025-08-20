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
import Levenshtein

#%% Set file directory
data_file = '../../'

#%%
#data = pd.read_csv('D:/PhD thesis/GCN/My Code/data/gan_a0201.csv')
data = pd.read_csv(data_file + 'data/neoepitopes/Bladder.4.0_test_mut.csv')
raw = data['peptide'].values
# from difflib import SequenceMatcher
# def similar(a,b):
#     return SequenceMatcher(None,a,b).ratio()
def editdistance(a,b):
    return Levenshtein.distance(a, b)

"""
Test peptides data
"""
# method = 'GAN_RL_old'
# method = 'GAN_RL_imm_select_LG_mean_only'
method_array = ["WGAN-GP",
                "GD_WGAN_wclip_MolGAN_DScalar_Loss_MyPredictor",
                "GD_WGAN_wclip_MolGAN_Gamma0_DScalar_Loss_MyPredictor",
                "GD_WGAN_wclip_OR_MolGAN_Gamma0_DScalar_Loss_MyPredictor",
                "Goal-directed_WGAN-GP_Gamma0_25",
                "Goal-directed_WGAN-GP_Gamma0_5",
                "Goal-directed_WGAN-GP_Gamma0_75",
                "Goal-directed_WGAN-GP",
                "Goal-directed_WGAN-GP_ORGAN",
                "Goal-directed_WGAN-GP_noRewardNet",
                "Goal-directed_WGAN-GP_noSscale",
                ]

name_array = ["WGAN-GP",
              r"MolGAN ($\lambda_M=0.5$)",
              r"MolGAN ($\lambda_M=0$)",
              r"MolGAN ($\lambda_M=0$) with ORGAN",
              r"GD-WGAN-GP ($\gamma_{max}=0.25$)",
              r"GD-WGAN-GP ($\gamma_{max}=0.5$)",
              r"GD-WGAN-GP ($\gamma_{max}=0.75$)",
              r"GD-WGAN-GP ($\gamma_{max}=1$)",
              "GD_WGAN-GP with ORGAN",
              r"GD-WGAN-GP ($\gamma_{max}=1$) without reward network",
              r"GD-WGAN-GP ($\gamma_{max}=1$ and $S_{scale}=1$)",
              ]

epoch = 1000
num_epochs = 1000
batch_size = 10000 # 6232 peptides = len(raw_Bladder)
ep = 'epoch' + str(epoch)
# data_path = 'D:/PhD thesis/GCN/My Code/' + method + '/' + ep + '/deepimmuno-GANRL-bladder-' + ep + '_rmv.txt'
for method, method_name in zip(method_array, name_array):
    data_path = data_file + 'result/' + method + '/' + ep + '/deepimmuno-GANRL-bladder-epoch' + str(num_epochs) + '-batch' + str(batch_size) + '_rmv.txt'
    generate = pd.read_csv(data_path,sep='\t')
    outdir = data_file + 'result/' + method + '/' + ep + '/'
    
    #generate = pd.read_csv('D:/PhD thesis/GCN/My Code/data/df/df_all_epoch100.csv')
    #generate = pd.read_csv('D:/PhD thesis/GCN/My Code/Test/GAN_RL_no_label_result/epoch0/test_generator_0.csv')
    seq = generate['peptide'].values
    
    len_seq = len(seq)
    count = 0
    all_value = []
    whole1 = []  # store mean value
    whole2 = []  # store max value
    for idx1, item1 in enumerate(seq):
        total = []
        for item2 in seq[idx1:]:
            each_edit = editdistance(item1, item2)
            total.append((item2, each_edit))
            all_value.append(each_edit)
        total = np.array(sorted(total, reverse=True, key=lambda x: x[1]))[:,1].astype(np.float64)
        # total2 = sorted(total,reverse=True,key=lambda x:x[1])[:5]
        # print('{0}:{1}'.format(item1,total2),file=f)
        count = count + 1
        if not (count%1000):
            print(str(count)+'/'+str(len_seq))
        whole1.append(total.mean())
        whole2.append(total.min())
    
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    mpl.rcParams['pdf.fonttype'] = 42
    mpl.rcParams['ps.fonttype'] = 42
    mpl.rcParams['font.family'] = 'Arial'
    
    # import seaborn as sns
    # #increase font size of all elements
    # sns.reset_defaults()
    # #sns.set(font_scale=1.0)
    # plot = sns.displot(data=pd.DataFrame({'Min. Edit Distance': whole2}),x='Min. Edit Distance',kde=True, aspect=1.5)
    # plt.grid(b=True,axis='y',alpha=0.3)
    # #plt.xticks(np.arange(0.3, 0.05, 1))
    # # plot.fig.set_figwidth(5)
    # # plot.fig.set_figheight(10)
    # plot.set_xticklabels(size = 15)
    # plot.set_yticklabels(size = 15)
    # plt.title('Min. Pairwise Edit Distance'+ str(num_epochs), fontsize=20)
    # plt.xlabel('Min. Edit Distance',fontsize=20);
    # plt.ylabel('Count',fontsize=20);
    # plt.savefig(outdir + '/sns_EditDistance_epoch' + str(num_epochs) + '.png')
    # plt.show()
    
    plt.rcParams.update({'font.size': 15}) # must set in top
    df=pd.DataFrame({'Edit Distance': all_value})
    num = df['Edit Distance'].unique()
    hist = df.hist(bins=len(num), color='#86bf91', edgecolor = 'black' , rwidth=0.5, figsize=(8,6), density=True)  
    plt.grid(visible=True,axis='y',alpha=0.3)
    plt.title(method_name, fontsize=20)
    plt.xlabel('Edit Distance',fontsize=20)
    plt.ylabel('Probability',fontsize=20)
    # Since only one column, take the first Axes
    ax = hist[0,0]
    # Add numbers on top of bars
    for patch in ax.patches:
        height = patch.get_height() #* 100   # convert from density to %
        if height > 0:
            ax.text(
                patch.get_x() + patch.get_width() / 2,
                patch.get_height(),
                f'{height:.2f}',
                ha='center', va='bottom', fontsize=12
            )
    # Add vertical dashed line for mean
    mean_val = np.mean(all_value)
    plt.axvline(mean_val, color='red', linestyle='--', linewidth=2)
    
    # Annotate mean value
    plt.text(
        mean_val, plt.ylim()[1]*0.9,         # position near top
        f'Mean = {mean_val:.2f}',
        color='red', fontsize=14, ha='center', va='bottom',
        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none')  # background box
    )
    plt.savefig(outdir + '/df_EditDistance_epoch' + str(num_epochs) + '.png')
    plt.show()
    
    
    print("outdir is {}".format(outdir))
    # plot.savefig(outdir + '/EditDistance_epoch' + str(num_epochs) + '.pdf')
    # np.sum(np.where(np.array(whole2) > 0.75, 1,0))  # 891- 60%
    
    #np.save(outdir + '/MaxSimilarity_' + ep  + '-batch' + str(batch_size), whole2)
    np.save(outdir + '/EditDistance_epoch' + str(num_epochs) + '-batch' + str(batch_size) + '_all_value', all_value)
    np.save(outdir + '/EditDistance_epoch' + str(num_epochs) + '-batch' + str(batch_size) + '_whole1', whole1)
    np.save(outdir + '/EditDistance_epoch' + str(num_epochs) + '-batch' + str(batch_size) + '_whole2', whole2)
    
    # count = 0
    # #with open('D:/PhD thesis/GCN/My Code/' + method + '/' + ep + '/log_rmv_0.txt','w') as f:
    # with open(data_file + 'result/' + method + '/' + ep + '/epoch' + str(num_epochs) + '-batch' + str(batch_size) +'log_rmv_0.txt','w') as f:
    #     for item1 in seq:
    #         total = []
    #         for item2 in raw:
    #             total.append((item2,similar(item1,item2)))
    #         total = sorted(total,reverse=True,key=lambda x:x[1])[:5]
    #         print('{0}:{1}'.format(item1,total),file=f)
    #         count = count + 1
    #         if not (count%10):
    #             print(str(count)+'/'+str(len_seq))
    
    print(f'\n Method: {method}')
    # print(f"Number of same peptides: {whole2.count(0)}")