# -*- coding: utf-8 -*-
"""
Created on Fri Mar 31 09:10:06 2023

@author: ych22001
"""

# sequence matching
import pandas as pd
import numpy as np

#%% Set file directory
data_file = '../../'

#%%
# method = 'GAN_RL_imm_select_LG_mean_only'
method_array = ["GD_WGAN_wclip_MolGAN_DScalar_Loss_MyPredictor",
                "GD_WGAN_wclip_MolGAN_Gamma0_DScalar_Loss_MyPredictor",
                "GD_WGAN_wclip_OR_MolGAN_Gamma0_DScalar_Loss_MyPredictor"]

#data = pd.read_csv('D:/PhD thesis/GCN/My Code/data/gan_a0201.csv')
# data = pd.read_csv('D:/PhD thesis/GCN/My Code/Test/Database/Bladder.4.0/Bladder.4.0_test_mut.csv')
# raw = data['peptide'].values
# from difflib import SequenceMatcher
# def similar(a,b):
#     return SequenceMatcher(None,a,b).ratio()

"""
Test peptides data
"""
# ep = 'epoch0'
# data_path = data_file + 'data/In paper/' + method + '/' + ep + '/NetMHCpan_prediction_' + ep + '.csv'
# prediction = pd.read_csv(data_path)

# #generate = pd.read_csv('D:/PhD thesis/GCN/My Code/data/df/df_all_epoch100.csv')
# #generate = pd.read_csv('D:/PhD thesis/GCN/My Code/Test/GAN_RL_no_label_result/epoch0/test_generator_0.csv')
# L_df0 = prediction['Aff(nM)'].loc[prediction['Aff(nM)']>=500]
# M_df0 = prediction['Aff(nM)'].loc[(prediction['Aff(nM)']<500) & (prediction['Aff(nM)']>=150)]
# S_df0 = prediction['Aff(nM)'].loc[prediction['Aff(nM)']<150]

# CountL_0 = L_df0.size
# CountM_0 = M_df0.size
# CountS_0 = S_df0.size

# ep = 'epoch20'
# data_path = data_file + 'data/In paper/' + method + '/' + ep + '/NetMHCpan_prediction_' + ep + '.csv'
# prediction = pd.read_csv(data_path)

# #generate = pd.read_csv('D:/PhD thesis/GCN/My Code/data/df/df_all_epoch100.csv')
# #generate = pd.read_csv('D:/PhD thesis/GCN/My Code/Test/GAN_RL_no_label_result/epoch0/test_generator_0.csv')
# L_df20 = prediction['Aff(nM)'].loc[prediction['Aff(nM)']>=500]
# M_df20 = prediction['Aff(nM)'].loc[(prediction['Aff(nM)']<500) & (prediction['Aff(nM)']>=150)]
# S_df20 = prediction['Aff(nM)'].loc[prediction['Aff(nM)']<150]

# CountL_20 = L_df20.size
# CountM_20 = M_df20.size
# CountS_20 = S_df20.size

# ep = 'epoch40'
# data_path = data_file + 'data/In paper/' + method + '/' + ep + '/NetMHCpan_prediction_' + ep + '.csv'
# prediction = pd.read_csv(data_path)

# #generate = pd.read_csv('D:/PhD thesis/GCN/My Code/data/df/df_all_epoch100.csv')
# #generate = pd.read_csv('D:/PhD thesis/GCN/My Code/Test/GAN_RL_no_label_result/epoch0/test_generator_0.csv')
# L_df40 = prediction['Aff(nM)'].loc[prediction['Aff(nM)']>=500]
# M_df40 = prediction['Aff(nM)'].loc[(prediction['Aff(nM)']<500) & (prediction['Aff(nM)']>=150)]
# S_df40 = prediction['Aff(nM)'].loc[prediction['Aff(nM)']<150]

# CountL_40 = L_df40.size
# CountM_40 = M_df40.size
# CountS_40 = S_df40.size

# ep = 'epoch60'
# data_path = data_file + 'data/In paper/' + method + '/' + ep + '/NetMHCpan_prediction_' + ep + '.csv'
# prediction = pd.read_csv(data_path)

# #generate = pd.read_csv('D:/PhD thesis/GCN/My Code/data/df/df_all_epoch100.csv')
# #generate = pd.read_csv('D:/PhD thesis/GCN/My Code/Test/GAN_RL_no_label_result/epoch0/test_generator_0.csv')
# L_df60 = prediction['Aff(nM)'].loc[prediction['Aff(nM)']>=500]
# M_df60 = prediction['Aff(nM)'].loc[(prediction['Aff(nM)']<500) & (prediction['Aff(nM)']>=150)]
# S_df60 = prediction['Aff(nM)'].loc[prediction['Aff(nM)']<150]

# CountL_60 = L_df60.size
# CountM_60 = M_df60.size
# CountS_60 = S_df60.size

# ep = 'epoch80'
# data_path = data_file + 'data/In paper/' + method + '/' + ep + '/NetMHCpan_prediction_' + ep + '.csv'
# prediction = pd.read_csv(data_path)

# #generate = pd.read_csv('D:/PhD thesis/GCN/My Code/data/df/df_all_epoch100.csv')
# #generate = pd.read_csv('D:/PhD thesis/GCN/My Code/Test/GAN_RL_no_label_result/epoch0/test_generator_0.csv')
# L_df80 = prediction['Aff(nM)'].loc[prediction['Aff(nM)']>=500]
# M_df80 = prediction['Aff(nM)'].loc[(prediction['Aff(nM)']<500) & (prediction['Aff(nM)']>=150)]
# S_df80 = prediction['Aff(nM)'].loc[prediction['Aff(nM)']<150]

# CountL_80 = L_df80.size
# CountM_80 = M_df80.size
# CountS_80 = S_df80.size

num_epochs_array = [816,6,6] # generator trained epoch

for method, num_epochs in zip(method_array, num_epochs_array):
    ep = 'epoch1000'
    data_path = data_file + 'result_brain/' + method + '/' + ep + '/best_NetMHCpan_prediction_epoch1000' + '.csv'
    prediction = pd.read_csv(data_path)
    
    #generate = pd.read_csv('D:/PhD thesis/GCN/My Code/data/df/df_all_epoch100.csv')
    #generate = pd.read_csv('D:/PhD thesis/GCN/My Code/Test/GAN_RL_no_label_result/epoch0/test_generator_0.csv')
    L_df1000 = prediction['Aff(nM)'].loc[prediction['Aff(nM)']>=500]
    M_df1000 = prediction['Aff(nM)'].loc[(prediction['Aff(nM)']<500) & (prediction['Aff(nM)']>=150)]
    S_df1000 = prediction['Aff(nM)'].loc[prediction['Aff(nM)']<150]
    
    CountL_1000 = L_df1000.size
    CountM_1000 = M_df1000.size
    CountS_1000 = S_df1000.size
    
    IC50 = ("IC$_{50}$<150", "150$\leq$IC$_{50}$<500", "500$\leq$IC$_{50}$")
    penguin_means = {
        # 'Epoch0': (CountS_0, CountM_0, CountL_0),
        # 'Epoch20': (CountS_20, CountM_20, CountL_20),
        # 'Epoch40': (CountS_40, CountM_40, CountL_40),
        # 'Epoch60': (CountS_60, CountM_60, CountL_60),
        # 'Epoch80': (CountS_80, CountM_80, CountL_80),
        'Epoch1000': (CountS_1000, CountM_1000, CountL_1000),
    }
    
    import matplotlib.pyplot as plt
    x = np.arange(len(IC50))*1.3  # the label locations
    width = 0.16  # the width of the bars
    multiplier = 0
    color=['#045F5F', '#00827F', '#5F9EA0', '#20B2AA', '#46C7C7', '#40E0D0']
    
    fig, ax = plt.subplots(constrained_layout=True, figsize=(12,5))
    #plt.figure(figsize=(20,8))
    
    # def autolabel(rects):
    #     """
    #     Attach a text label above each bar displaying its height
    #     """
    #     for rect in rects:
    #        height = rect.get_height()
    #        ax.text(rect.get_x() + rect.get_width()/2.-.01, 1.05*height,
    #                 '%d' % int(height),
    #        ha='center', va='bottom', fontsize=14)
    
    for attribute, measurement in penguin_means.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, measurement, width, label=attribute, color=color[multiplier])
        #autolabel(rects)
        #ax.text(rects.get_x()+rects.get_width()/2.,1.05*rects.get_height(),'%d' % int(rects.get_height()))
        ax.bar_label(rects, padding=3, fontsize=14)
        multiplier += 1
    
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_xlabel('The range of IC$_{50}$', weight='bold', fontsize=20)
    ax.set_ylabel('The number of peptides', weight='bold', fontsize=20)
    #ax.set_title('Penguin attributes by species')
    ax.set_xticks(x + width*2.5, IC50, fontsize=20)
    ax.yaxis.set_tick_params(labelsize=14)
    ax.legend(ncol=6, fontsize=14, bbox_to_anchor=(0.95, 1.15))
    #ax.set_ylim(0, 1024)
    ax.set_ylim(0, 1100)
    #fig.set_size_inches(18.5, 10.5)
    plt.savefig(data_file + 'result_brain/' + method + '/' + ep + "/best_IC50 compare.png", bbox_inches='tight')
    plt.show()
    
    print(f'Method: {method}')
    print(f"Number of peptide sequences with IC$_{50}$ < 500: {CountM_1000 + CountS_1000}")

# #--------------------------------------------------------------------------------------#
# epochs = ("0", "20", "40", "60", "80", "100")
# IC50_sums = {
#     'IC$_{50}$<500': (CountS_0+CountM_0, CountS_20+CountM_20, CountS_40+CountM_40, CountS_60+CountM_60, CountS_80+CountM_80, CountS_100+CountM_100),
#     'Immunogenic peptides': (514, 797, 716, 935, 959, 958),
# }

# x2 = np.arange(len(epochs))*0.65  # the label locations
# width2 = 5  # the width of the bars
# multiplier2 = 0
# color=['#89C35C', '#872657']

# fig, ax = plt.subplots(constrained_layout=True, figsize=(7,5))
# #plt.figure(figsize=(20,8))

# # plt.plot(np.array([0,20,40,60,80,100])*0.65,IC50_sums['IC$_{50}$<500'],'o-',color = '#A9A9A9',linestyle = 'dashed', alpha=0.7)
# # plt.plot(np.array([0,20,40,60,80,100])*0.65+width2,IC50_sums['Immunogenic \npeptides'],'o-',color = '#A9A9A9',linestyle = 'dashed', alpha=0.7)

# #plt.plot(np.array([0,20,40,60,80,100])*0.65,IC50_sums['IC$_{50}$<500'],'o-',color = '#008B8B',linestyle = 'dashed', alpha=0.5)
# #plt.plot(np.array([0,20,40,60,80,100])*0.65+width2,IC50_sums['Immunogenic \npeptides'],'o-',color = '#E56717',linestyle = 'dashed', alpha=0.5)

# for attribute, measurement in IC50_sums.items():
#     offset = width2 * multiplier2
#     rects = ax.bar(x2 * 20 + offset, measurement, width2, label=attribute, color=color[multiplier2])
#     ax.bar_label(rects, padding=3, fontsize=13)
#     multiplier2 += 1

# # Add some text for labels, title and custom x-axis tick labels, etc.
# ax.set_xlabel('Training epoch', weight='bold', fontsize=20)
# ax.set_ylabel('The number of peptides', weight='bold', fontsize=20)
# #ax.set_title('Penguin attributes by species')
# ax.set_xticks(x2*20, epochs, fontsize=20)
# ax.yaxis.set_tick_params(labelsize=20)
# ax.legend(ncol=2, fontsize=14, bbox_to_anchor=(0.85, 1.15))
# #ax.set_ylim(0, 1024)
# ax.set_ylim(0, 1100)
# #fig.set_size_inches(18.5, 10.5)
# plt.savefig(data_file + "results/(In paper) IC50 Imm.png", bbox_inches='tight')
# plt.show()