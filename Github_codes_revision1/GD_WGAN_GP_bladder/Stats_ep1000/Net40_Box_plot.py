# -*- coding: utf-8 -*-

# sequence matching
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#%% 1
method_array = ["Dataset",
                "random_generator",
                "WGAN-GP",
                "GD_WGAN_wclip_MolGAN_DScalar_Loss_MyPredictor",
                "GD_WGAN_wclip_MolGAN_DScalar_Loss_MyPredictor_best",
                "GD_WGAN_wclip_MolGAN_Gamma0_DScalar_Loss_MyPredictor",
                "GD_WGAN_wclip_MolGAN_Gamma0_DScalar_Loss_MyPredictor_best",
                "GD_WGAN_wclip_OR_MolGAN_Gamma0_DScalar_Loss_MyPredictor",
                "GD_WGAN_wclip_OR_MolGAN_Gamma0_DScalar_Loss_MyPredictor_best",
                "Goal-directed_WGAN-GP_Gamma0_25",
                "Goal-directed_WGAN-GP_Gamma0_5",
                "Goal-directed_WGAN-GP_Gamma0_75",
                "Goal-directed_WGAN-GP",
                "Goal-directed_WGAN-GP_ORGAN",
                "Goal-directed_WGAN-GP_noRewardNet",
                "Goal-directed_WGAN-GP_noSscale",
                "Goal-directed_WGAN-GP_NoSwithORGAN",
                "Goal-directed_WGAN-GP_LSTM",
                "Goal-directed_WGAN-GP_ORGAN_LSTM",
                "Goal-directed_WGAN-GP_ORGAN_TransformerNoMaskL2",
                "Goal-directed_WGAN-GP_TransformerNoMaskL2",
                                
                ]

name_array = ["Bladder dataset",
              "Random generator",
              "WGAN-GP",
              r"MolGAN ($\lambda_M=0.5$)",
              r"MolGAN best ($\lambda_M=0.5$)",
              r"MolGAN ($\lambda_M=0$)",
              r"MolGAN best ($\lambda_M=0$)",
              r"MolGAN ($\lambda_M=0$) with ORGAN",
              r"MolGAN best ($\lambda_M=0$) with ORGAN",
              r"GD-WGAN-GP ($\gamma_{max}=0.25$)",
              r"GD-WGAN-GP ($\gamma_{max}=0.5$)",
              r"GD-WGAN-GP ($\gamma_{max}=0.75$)",
              r"GD-WGAN-GP ($\gamma_{max}=1$)",
              "GD-WGAN-GP with ORGAN",
              r"GD-WGAN-GP ($\gamma_{max}=1$) without reward network",
              r"GD-WGAN-GP ($\gamma_{max}=1$ and $S_{scale}=1$)",
              r"GD-WGAN-GP without switch and ORGAN",
              r"GD-WGAN-GP ($\gamma_{max}=1$) and LSTM",
              "GD-WGAN-GP with ORGAN and LSTM",
              r"GD-WGAN-GP ($\gamma_{max}=1$) and Transformer",
              "GD-WGAN-GP with ORGAN and Transformer",
              ]
epoch = 1000
num_epochs = 1000
batch_size = 10000 # 6232 peptides = len(raw_Bladder)
ep = 'epoch' + str(epoch)
data_file = '../'
df = []
# data_path = 'D:/PhD thesis/GCN/My Code/' + method + '/' + ep + '/deepimmuno-GANRL-bladder-' + ep + '_rmv.txt'
output_file = "Net40results.txt"

with open(output_file, "w") as f:  # open file once before loop
    for method, method_name in zip(method_array, name_array):
        if "dataset" in method_name:
            data_path = data_file + 'Results/' + method + '/NetMHCpan_prediction_epoch1000.csv'
        elif "_best" in method:
            data_path = data_file + 'Results/' + method.replace("_best","") + '/' + ep + '/NetMHCpan_prediction_epoch1000.csv'
        else:    
            data_path = data_file + 'Results/' + method + '/' + ep + '/NetMHCpan_prediction_epoch1000.csv'
        generate = pd.read_csv(data_path)#,sep='\t')
        outdir = data_file + 'Results/' + method + '/' + ep + '/'
        ba = generate['Aff(nM)'].values
        seq = generate['Peptide'].values
        df_labled = pd.DataFrame({'Binding affinity score':ba})
        df_labled['Method'] = method_name
        df.append(df_labled)
        # Prepare the report text
        report = (
            f"\nMethod: {method}\n"
            f"# of sequences: {len(ba)}\n"
            f"Average score: {float(sum(ba)/len(ba))}\n"
            f"Std score: {float(np.std(ba))}\n"
            f"Median score: {float(np.median(ba))}\n"
            f"Min. score: {float(min(ba))}\n"
            f"Max. score: {float(max(ba))}\n"
        )

        # # Print to console
        # print(report)

        # Save to file
        f.write(report)
df_concate = pd.concat(df)

# #%%

# """
# No predictor peptides data 
# """
# ep = 'epoch1000'
# batch_size = 10000 # 10000 peptides
# method = 'WGAN-GP'

# data_path = method + '/' + 'VGAN_result.csv'
# generate = pd.read_csv(data_path)
# imm = generate['score'].values

# seq = generate['peptide'].values
# WGANGP_seq= seq

# df = pd.DataFrame({'immunogenicity':imm})

# """
# With predictor peptides data 
# """
# ep = 'epoch1000'
# batch_size = 10000 # 10000 peptides
# method = 'Goal-directed_WGAN-GP'

# data_path = method + '/' + 'GGAN_result.csv'
# generate = pd.read_csv(data_path)
# imm = generate['score'].values

# seq = generate['peptide'].values
# GD_seq = seq

# df2 = pd.DataFrame({'immunogenicity':imm})

# """
# Combine two dataframe
# """

# df_labled = df
# df_labled['Method'] = 'WGAN-GP'
# df2_labled = df2
# df2_labled['Method'] = 'Goal-directed WGAN-GP'
# df_concate = pd.concat([df2_labled, df_labled])
# df_concate['Immunogenicity score'] = df_concate['immunogenicity']

#%% Box plot
color_array = ['#0173b2',
               '#0173b2',
               '#0173b2',
               '#de8f05',
               '#de8f05',
               '#de8f05',
               '#de8f05',
               '#de8f05',
               '#de8f05',
               '#029e73',
               '#029e73',
               '#029e73',
               '#029e73',
               '#029e73',
               '#cc78bc',
               '#cc78bc',
               '#cc78bc',
               '#949494',
               '#949494',
               '#949494',
               '#949494',]
plt.rcParams['figure.figsize']
# [6.4, 4.8]
# print(sns.color_palette().as_hex())
# blue: #1f77b4
# orange: #ff7f0e
fig3, ax3 = plt.subplots(figsize=(10, 10))
sns.boxplot(data=df_concate, x="Method", y="Binding affinity score",saturation=0.5,width=0.5,palette=color_array, ax=ax3)
sns.stripplot(data=df_concate, x="Method", y="Binding affinity score",palette=color_array, size=1, alpha=0.5, ax=ax3)
# iterate through the axes containers
for c in ax3.xaxis.get_major_ticks():
    c.label.set_fontsize(15)
    c.label.set_rotation(90)
for c in ax3.yaxis.get_major_ticks():
    c.label.set_fontsize(15)
plt.title('NetMHCpan4.0',fontsize=25, weight='bold')
plt.ylabel('Binding affinity score',fontsize=20, weight='bold')
plt.xlabel('Methods',fontsize=20, weight='bold')
plt.yscale('log')  # Set y-axis to logarithmic scale
plt.grid()
plt.tight_layout()
# Save the plot with a specified resolution (e.g., 300 DPI)
plt.savefig('Net40_box.png', dpi=300)
plt.show()

# #%% Count unique sequence
# # our generated peptides
# g_unq_rows, g_count = np.unique(GD_seq,return_counts=1)
# g_num_unq = len(g_count)
# print('# of unique peptides from our method: ' + str(g_num_unq))
# print('Most frequent sequence: ' + str(g_unq_rows[np.argmax(g_count)] + ', ' + str(np.max(g_count)) + ' occurance'))

# # others generated peptides
# w_unq_rows, w_count = np.unique(WGANGP_seq,return_counts=1)
# w_num_unq = len(w_count)
# print('# of unique peptides from other method: ' + str(w_num_unq))
# print('Most frequent sequence: ' + str(w_unq_rows[np.argmax(w_count)] + ', ' + str(np.max(w_count)) + ' occurance'))
