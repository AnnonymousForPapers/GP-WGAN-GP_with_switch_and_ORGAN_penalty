# -*- coding: utf-8 -*-
'''
Revised from https://github.com/frankligy/DeepImmuno
DeepImmuno: deep learning-empowered prediction and generation of 
immunogenic peptides for T-cell immunity, Briefings in Bioinformatics, 
May 03 2021 (https://doi.org/10.1093/bib/bbab160)
'''
"""
Created on Wed Feb 15 08:33:05 2023

@author: ych22001
python 3.9.15
"""

import tensorflow.keras as keras
from tensorflow.keras import layers

#%% Set file directory
data_file = '../../'

#%% CNN model
input1 = keras.Input(shape=(10, 12, 1)) # TensorShape([None, 10, 12, 1])
input2 = keras.Input(shape=(46, 12, 1))

# 16*2*12 parameters
x = layers.Conv2D(filters=16, kernel_size=(2, 12))(input1)  # 9 
# x.shape = TensorShape([None, 9, 1, 16])
x = layers.BatchNormalization()(x)
x = keras.activations.relu(x)
x = layers.Conv2D(filters=32, kernel_size=(2, 1))(x)    # 8
# x.shape = TensorShape([None, 8, 1, 32])
x = layers.BatchNormalization()(x)
x = keras.activations.relu(x)
x = layers.MaxPool2D(pool_size=(2, 1), strides=(2, 1))(x)  # 4
# x.shape = TensorShape([None, 4, 1, 32])
x = layers.Flatten()(x)
# x.shape = TensorShape([None, 128])
x = keras.Model(inputs=input1, outputs=x)

y = layers.Conv2D(filters=16, kernel_size=(15, 12))(input2)     # 32
y = layers.BatchNormalization()(y)
y = keras.activations.relu(y)
y = layers.MaxPool2D(pool_size=(2, 1), strides=(2, 1))(y)  # 16
y = layers.Conv2D(filters=32,kernel_size=(9,1))(y)    # 8
y = layers.BatchNormalization()(y)
y = keras.activations.relu(y)
y = layers.MaxPool2D(pool_size=(2, 1),strides=(2,1))(y)  # 4
y = layers.Flatten()(y)
y = keras.Model(inputs=input2,outputs=y)

combined = layers.concatenate([x.output,y.output])
z = layers.Dense(128,activation='relu')(combined)
z = layers.Dropout(0.2)(z)
z = layers.Dense(1,activation='sigmoid')(z)

cnn_model = keras.Model(inputs=[input1,input2],outputs=z)
# https://www.tensorflow.org/tutorials/keras/save_and_load
cnn_model.load_weights(data_file + 'weights/Immunogenicity_Predictor/')
#cnn_model.load_weights('C:/Users/ych22001/OneDrive - University of Connecticut/Documents/2. GCN/Vaccine/Code/My Code-20230118T182446Z-001/My Code/training_1/')
# end of CNN model

import numpy as np
import pandas as pd
import os

#%%
method_array = ["GD_WGAN_wclip_MolGAN_DScalar_Loss_MyPredictor",
                "GD_WGAN_wclip_MolGAN_Gamma0_DScalar_Loss_MyPredictor",
                "GD_WGAN_wclip_OR_MolGAN_Gamma0_DScalar_Loss_MyPredictor"]
# method = 'GAN_RL_old'
# method = 'GAN_RL_imm_select_LG_mean_only'
ep = 'epoch1000'
# num_epochs = 1000
num_epochs_array = [816,6,6] # generator trained epoch
batch_size = 10000 # 6232 peptides = len(raw_Bladder)
for method, num_epochs in zip(method_array, num_epochs_array):
    outdir = data_file + 'result/' + method + '/' + ep + '/'
    print("outdir is {}".format(outdir))
    outname = 'best_deepimmuno-GANRL-bladder_cancer-scored_epoch' + str(num_epochs) + '-batch' + str(batch_size) + '.txt'
    """
    Test peptides data
    """
    #data_path = 'D:/PhD thesis/GCN/My Code/' + method + '/' + ep + '/deepimmuno-GANRL-bladder-' + ep + '_rmv.txt'
    data_path = data_file + 'result/' + method + '/' + ep + '/best_deepimmuno-GANRL-bladder-epoch' + str(num_epochs) + '-batch' + str(batch_size) + '_rmv.txt'
    df = pd.read_csv(data_path,sep='\t')
    
    #temp = []
    #for i in range(len(df['peptide'])):
    #     temp.append(1) 
    #df['immunogenicity'] = temp
    
    """
    Get immunogenicity using CNN
    """
    
    
    pseudo = df['peptide']
    epitope = pseudo
    mhc = 'HLA-A*0201'
    #score = computing_s(epitope,hla)
    # 556 amino acid associated physicochemical properties
    # chose 12 principal components
    after_pca = np.loadtxt(data_file + 'data/DeepImmuno/after_pca.txt') # (21,12)
    hla = pd.read_csv(data_file + 'data/DeepImmuno/hla2paratopeTable_aligned.txt',sep='\t')
    
    # hla_dic = hla_df_to_dic(hla)
    # assign each HLA a pseudo sequence and form a dictionary
    hla_dic = {}
    for i in range(hla.shape[0]): # (62,2) run 62 times
        col1 = hla['HLA'].iloc[i]  # HLA allele, take the ith row data in the HLA column
        col2 = hla['pseudo'].iloc[i]  # pseudo sequence
        hla_dic[col1] = col2
        
    inventory = list(hla_dic.keys()) # save all the HLA 
    
    #dic_inventory = dict_inventory(inventory)
    dicA, dicB, dicC = {}, {}, {}
    dic_inventory = {'A': dicA, 'B': dicB, 'C': dicC}
    
    # For all the hla in inventory,
    # make dictionary according to gene type (locus) and the allele group (first2)
    # the last2 digit represents a specific HLA protein
    #ex: HLA-C*1510
    for hla_i in inventory:
        type_ = hla_i[4]  # A,B,C take C
        first2 = hla_i[6:8]  # 01 take 15
        last2 = hla_i[8:]  # 01 take 10
        try:
            dic_inventory[type_][first2].append(last2)
        except KeyError: # Assign space for the corresponding allele group (first2)
            dic_inventory[type_][first2] = []
            dic_inventory[type_][first2].append(last2)
            
    
    peptide_score = [epitope]
    hla_score = [mhc]
    immuno_score = ['0']
    # construct dataframe
    # https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html
    ori_score = df
    
    #dataset_score = construct_aaindex(ori_score,hla_dic,after_pca,dic_inventory)
    # Assign the 12 physicochemical properties to each amino acid
    # for peptide (10) and HLA (46)
    # and set the initial immunogenecity score to 0
    series = []
    # For all input epitope and HLA set
    for i in range(ori_score.shape[0]):
        peptide = ori_score['peptide'].iloc[i]
        hla_type = ori_score['HLA'].iloc[i]
        # reshape immunogenicity into 2d
        # https://stackoverflow.com/questions/18691084/what-does-1-mean-in-numpy-reshape
        immuno = np.array(ori_score['immunogenicity'].iloc[i]).reshape(1,-1)   # [1,1]
    
        #encode_pep = peptide_data_aaindex(peptide,after_pca)    # [10,12]
        # Assign the 12 physicochemical properties to each amino acid (10) in the peptide
        # According to the length of the peptide,
        # If it is 10, 
        # return numpy array [10,12,1]
        length = len(peptide)
        if length == 10:
            #encode = aaindex(peptide,after_pca)
            amino = 'ARNDCQEGHILKMFPSTWYV-'
            matrix = np.transpose(after_pca)   # [12,21]
            encode_pep = np.empty([len(peptide), 12])  # (seq_len,12)
            for i in range(len(peptide)):
                query = peptide[i]
                if query == 'X': query = '-' # maybe its checking the incorrect character?
                query = query.upper() # convert lowercase character to upper case
                encode_pep[i, :] = matrix[:, amino.index(query)] # the position of query
        elif length == 9:
            peptide = peptide[:5] + '-' + peptide[5:]
            #encode = aaindex(peptide,after_pca)
            amino = 'ARNDCQEGHILKMFPSTWYV-'
            matrix = np.transpose(after_pca)   # [12,21]
            encode_pep = np.empty([len(peptide), 12])  # (seq_len,12)
            for i in range(len(peptide)):
                query = peptide[i]
                if query == 'X': query = '-' # maybe its checking the incorrect character?
                query = query.upper() # convert lowercase character to upper case
                encode_pep[i, :] = matrix[:, amino.index(query)] # the position of query
        encode_pep = encode_pep.reshape(encode_pep.shape[0], encode_pep.shape[1], -1)
        # end of encode_pep = peptide_data_aaindex(peptide,after_pca)
        
        #encode_hla = hla_data_aaindex(hla_dic,hla_type,after_pca,dic_inventory)   # [46,12]
        # Assign the 12 physicochemical properties to each amino acid (46) in the hla
        # return numpy array [46,12,1]
        try:
            seq = hla_dic[hla_type]
        except KeyError:
            # For unknown hla sequence
            #hla_type = rescue_unknown_hla(hla_type,dic_inventory)
            type_ = hla[4]
            first2 = hla[6:8]
            last2 = hla[8:]
            big_category = dic_inventory[type_]
            #print(hla)
            if not big_category.get(first2) == None:
                small_category = big_category.get(first2)
                distance = [abs(int(last2) - int(i)) for i in small_category]
                optimal = min(zip(small_category, distance), key=lambda x: x[1])[0]
                hla_type = 'HLA-' + str(type_) + '*' + str(first2) + str(optimal)
            else:
                small_category = list(big_category.keys())
                distance = [abs(int(first2) - int(i)) for i in small_category]
                optimal = min(zip(small_category, distance), key=lambda x: x[1])[0]
                hla_type = 'HLA-' + str(type_) + '*' + str(optimal) + str(big_category[optimal][0])
            seq = hla_dic[hla_type]
        #encode = aaindex(seq,after_pca)
        amino = 'ARNDCQEGHILKMFPSTWYV-'
        matrix = np.transpose(after_pca)   # [12,21]
        encode_hla = np.empty([len(seq), 12])  # (seq_len,12)
        for i in range(len(seq)):
            query = seq[i]
            if query == 'X': query = '-'
            query = query.upper() # convert lowercase character to upper case
            encode_hla[i, :] = matrix[:, amino.index(query)] # the position of query
        
        encode_hla = encode_hla.reshape(encode_hla.shape[0], encode_hla.shape[1], -1)
        # end of encode_hla = hla_data_aaindex(hla_dic,hla_type,after_pca,dic_inventory)
        
        series.append((encode_pep, encode_hla, immuno))
        dataset_score = series
        # end of def construct_aaindex(ori,hla_dic,after_pca,dic_inventory):
            
    #input1_score = pull_peptide_aaindex(dataset_score)
    # For each input peptide,
    # store their pca values for each amino acid
    # the same as encode_pep? No
    # input1_score has its first index for each input peptide
    input1_score = np.empty([len(dataset_score),10,12,1])
    for i in range(len(dataset_score)):
        input1_score[i,:,:,:] = dataset_score[i][0]
    # end of def pull_peptide_aaindex(dataset):
    
    #input2_score = pull_hla_aaindex(dataset_score)
    input2_score = np.empty([len(dataset_score),46,12,1])
    for i in range(len(dataset_score)):
        input2_score[i,:,:,:] = dataset_score[i][1]
    # end of def pull_hla_aaindex(dataset):
    
    #label_score = pull_label_aaindex(dataset_score)
    label_score = np.empty([len(dataset_score),1])
    for i in range(len(dataset_score)):
        label_score[i,:] = dataset_score[i][2]
    #end of def pull_label_aaindex(dataset):
    
    # for epitope = 'PLRTQWERN'
    # hla = 'HLA-A*0201'
    scoring = cnn_model.predict(x=[input1_score,input2_score])
    df = pd.DataFrame({'peptide': pseudo, 'HLA': ['HLA-A*0201' for i in range(len(pseudo))],
                       'immunogenicity': [float(scoring[i]) for i in range(len(pseudo))]})
    df.to_csv(os.path.join(outdir,outname),sep='\t',index=None)
    # print(sum(scoring>0.5))
    print(f'\n Method: {method}')
    print(f"# of sequences: {len(scoring)}")
    print(f"Average score: {float(sum(scoring)/len(scoring))}")
    print(f"Std score: {float(np.std(scoring))}")
    print(f"Median score: {float(np.median(scoring))}")
    print(f"Min. score: {float(min(scoring))}")
    print(f"Max. score: {float(max(scoring))}")