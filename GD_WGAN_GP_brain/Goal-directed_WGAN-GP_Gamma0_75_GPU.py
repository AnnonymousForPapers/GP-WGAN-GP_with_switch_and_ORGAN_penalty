'''
Revised from https://github.com/frankligy/DeepImmuno
DeepImmuno: deep learning-empowered prediction and generation of 
immunogenic peptides for T-cell immunity, Briefings in Bioinformatics, 
May 03 2021 (https://doi.org/10.1093/bib/bbab160)
'''

import timeit
start_whole = timeit.default_timer()
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import random

random.seed(53)
random.random()

torch.manual_seed(53)
torch.rand(4)

np.random.seed(53)
np.random.rand(4)

def sigmoid(x):
  return 1 / (1 + torch.exp(-x))

# build the model
class ResBlock(nn.Module):
    def __init__(self,hidden):    # hidden means the number of filters
        super(ResBlock,self).__init__()
        self.res_block = nn.Sequential(
            nn.ReLU(True),    # in_place = True
            nn.Conv1d(hidden,hidden,kernel_size=3,padding=1),
            nn.ReLU(True),
            nn.Conv1d(hidden,hidden,kernel_size=3,padding=1),
        )

    def forward(self,input):   # input [N, hidden, seq_len]
        output = self.res_block(input)
        return input + 0.3*output   # [N, hidden, seq_len]  doesn't change anything

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
    
class Discriminator(nn.Module):
    def __init__(self,hidden,n_chars,seq_len):
        super(Discriminator,self).__init__()
        self.block = nn.Sequential(
            ResBlock(hidden),
            ResBlock(hidden),
            ResBlock(hidden),
            ResBlock(hidden),
            ResBlock(hidden),
        )
        self.conv1 = nn.Conv1d(n_chars,hidden,1)
        self.fc = nn.Linear(seq_len*hidden,1)
        self.hidden = hidden
        self.n_chars = n_chars
        self.seq_len = seq_len

    def forward(self,input):  # input [N,seq_len,n_chars]
        output = input.transpose(1,2)   # input [N, n_chars, seq_len]
        output = output.contiguous()
        output = self.conv1(output)  # [N,hidden,seq_len]
        output = self.block(output)  # [N, hidden, seq_len]
        output = output.view(-1,self.seq_len*self.hidden)  # [N, hidden*seq_len]
        output = self.fc(output)   # [N,1]
        return output
    
class Scorer(nn.Module):
    def __init__(self,hidden,n_chars,seq_len):
        super(Scorer,self).__init__()
        self.block = nn.Sequential(
            ResBlock(hidden),
            ResBlock(hidden),
            ResBlock(hidden),
            ResBlock(hidden),
            ResBlock(hidden),
        )
        self.conv1 = nn.Conv1d(n_chars,hidden,1)
        self.fc = nn.Linear(seq_len*hidden,1)
        self.hidden = hidden
        self.n_chars = n_chars
        self.seq_len = seq_len

    def forward(self,input):  # input [N,seq_len,n_chars]
        output = input.transpose(1,2)   # input [N, n_chars, seq_len]
        output = output.contiguous()
        output = self.conv1(output)  # [N,hidden,seq_len]
        output = self.block(output)  # [N, hidden, seq_len]
        output = output.view(-1,self.seq_len*self.hidden)  # [N, hidden*seq_len]
        output = self.fc(output)   # [N,1]
        return output

# define dataset
class real_dataset_class(torch.utils.data.Dataset):
    def __init__(self,raw,seq_len,n_chars):  # raw is a ndarray ['ARRRR','NNNNN']
        self.raw = raw
        self.seq_len = seq_len
        self.n_chars = n_chars
        self.post = self.process()


    def process(self):
        result = torch.empty(len(self.raw),self.seq_len,self.n_chars)   # [N,seq_len,n_chars]
        amino = 'ARNDCQEGHILKMFPSTWYV-'
        identity = torch.eye(n_chars)
        for i in range(len(self.raw)):
            pep = self.raw[i]
            if len(pep) == 9:
                pep = pep[0:4] + '-' + pep[4:]
            inner = torch.empty(len(pep),self.n_chars)
            for p in range(len(pep)):
                query = pep[p]
                if query == 'X':
                    query = '-'
                inner[p] = identity[amino.index(query.upper()), :]
            encode = torch.tensor(inner)   # [seq_len,n_chars]
            result[i] = encode
        return result


    def __getitem__(self,index):
        return self.post[index]

    def __len__(self):
        return self.post.shape[0]


# auxiliary function during training GAN
def sample_generator(batch_size):
    noise = torch.randn(batch_size,128).to(device)  # [N, 128]
    generated_data = G(noise)   # [N, seq_len, n_chars]
    return generated_data

def calculate_gradient_penalty(real_data,fake_data,lambda_=10):
    alpha = torch.rand(batch_size,1,1).to(device) # 64*1*1
    alpha = alpha.expand_as(real_data)   # [N,seq_len,n_chars] 64*10*21
    interpolates = alpha * real_data + (1-alpha) * fake_data  # [N,seq_len,n_chars]
    interpolates = torch.autograd.Variable(interpolates,requires_grad=True)
    disc_interpolates = D(interpolates)
    # below, grad function will return a tuple with length one, so only take [0], it will be a tensor of shape inputs, gradient wrt each input
    # d(output)/d(input)
    # 64*10*21
    gradients = torch.autograd.grad(outputs=disc_interpolates,inputs=interpolates,grad_outputs=torch.ones(disc_interpolates.size()).to(device),create_graph=True,retain_graph=True)[0]
    gradients = gradients.contiguous().view(batch_size,-1)  # [N, seq_len*n_chars] resize
    gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)  # [N,]
    gradient_penalty = lambda_* ((gradients_norm - 1) ** 2).mean()     # []
    return gradient_penalty


# processing function from previous code
def peptide_data_aaindex(peptide,after_pca):   # return numpy array [10,12,1]
    length = len(peptide)
    if length == 10:
        encode = aaindex(peptide,after_pca)
    elif length == 9:
        peptide = peptide[:5] + '-' + peptide[5:]
        encode = aaindex(peptide,after_pca)
    encode = encode.reshape(encode.shape[0], encode.shape[1], -1)
    return encode


def dict_inventory(inventory):
    dicA, dicB, dicC = {}, {}, {}
    dic = {'A': dicA, 'B': dicB, 'C': dicC}

    for hla in inventory:
        type_ = hla[4]  # A,B,C
        first2 = hla[6:8]  # 01
        last2 = hla[8:]  # 01
        try:
            dic[type_][first2].append(last2)
        except KeyError:
            dic[type_][first2] = []
            dic[type_][first2].append(last2)

    return dic


def rescue_unknown_hla(hla, dic_inventory):
    type_ = hla[4]
    first2 = hla[6:8]
    last2 = hla[8:]
    big_category = dic_inventory[type_]
    #print(hla)
    if not big_category.get(first2) == None:
        small_category = big_category.get(first2)
        distance = [abs(int(last2) - int(i)) for i in small_category]
        optimal = min(zip(small_category, distance), key=lambda x: x[1])[0]
        return 'HLA-' + str(type_) + '*' + str(first2) + str(optimal)
    else:
        small_category = list(big_category.keys())
        distance = [abs(int(first2) - int(i)) for i in small_category]
        optimal = min(zip(small_category, distance), key=lambda x: x[1])[0]
        return 'HLA-' + str(type_) + '*' + str(optimal) + str(big_category[optimal][0])



def hla_df_to_dic(hla):
    dic = {}
    for i in range(hla.shape[0]):
        col1 = hla['HLA'].iloc[i]  # HLA allele
        col2 = hla['pseudo'].iloc[i]  # pseudo sequence
        dic[col1] = col2
    return dic

def aaindex(peptide,after_pca):

    amino = 'ARNDCQEGHILKMFPSTWYV-'
    matrix = np.transpose(after_pca)   # [12,21]
    encoded = np.empty([len(peptide), 12])  # (seq_len,12)
    for i in range(len(peptide)):
        query = peptide[i]
        if query == 'X': query = '-'
        query = query.upper()
        encoded[i, :] = matrix[:, amino.index(query)]

    return encoded


# post utils functions
def inverse_transform(hard):   # [N,seq_len]
    amino = 'ARNDCQEGHILKMFPSTWYV-'
    result = []
    for row in hard:
        temp = ''
        for col in row:
            aa = amino[col]
            temp += aa
        result.append(temp)
    return result

import tensorflow.keras as keras
from tensorflow.keras import layers


def seperateCNN():
    input1 = keras.Input(shape=(10, 12, 1))
    input2 = keras.Input(shape=(46, 12, 1))

    x = layers.Conv2D(filters=16, kernel_size=(2, 12))(input1)  # 9
    x = layers.BatchNormalization()(x)
    x = keras.activations.relu(x)
    x = layers.Conv2D(filters=32, kernel_size=(2, 1))(x)    # 8
    x = layers.BatchNormalization()(x)
    x = keras.activations.relu(x)
    x = layers.MaxPool2D(pool_size=(2, 1), strides=(2, 1))(x)  # 4
    x = layers.Flatten()(x)
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

    model = keras.Model(inputs=[input1,input2],outputs=z)
    return model

def trainedCNN(after_pca, hla, cnn_model, generated_data):
    generation = generated_data.detach().cpu().numpy()
    # https://numpy.org/doc/stable/reference/generated/numpy.argmax.html
    # Returns the indices of the maximum values along an axis
    hard = np.argmax(generation, axis=2)  # [N,seq_len]
    pseudo = inverse_transform(hard)
    df = pd.DataFrame({'peptide': pseudo, 'HLA': ['HLA-A*0201' for i in range(len(pseudo))],
                       'immunogenicity': [1 for i in range(len(pseudo))]})

    epitope = pseudo
    mhc = 'HLA-A*0201'

    # assign each HLA a pseudo sequence and form a dictionary
    hla_dic = {}
    for i in range(hla.shape[0]): # (62,2) run 62 times
        col1 = hla['HLA'].iloc[i]  # HLA allele, take the ith row data in the HLA column
        col2 = hla['pseudo'].iloc[i]  # pseudo sequence
        hla_dic[col1] = col2
        
    inventory = list(hla_dic.keys()) # save all the HLA 
    
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
    
         # Assign the 12 physicochemical properties to each amino acid (10) in the peptide
        # According to the length of the peptide,
        # If it is 10, 
        # return numpy array [10,12,1]
        length = len(peptide)
        if length == 10:
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
    
    # https://www.tensorflow.org/api_docs/python/tf/keras/Model#predict
    scoring = cnn_model.predict(x=[input1_score,input2_score],verbose=0)
    # np.array to torch https://pytorch.org/docs/stable/tensors.html
    # Torch with gradient https://pytorch.org/tutorials/beginner/former_torchies/autograd_tutorial.html
    scoring_grad = torch.tensor(scoring,requires_grad=True)
    return scoring_grad

batch_size = 64
lr = 0.0001
num_epochs = 1000
seq_len = 10
hidden = 128
n_chars = 21
d_steps = 10
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')

data_file = '../'

# Load CNN model
cnn_model = seperateCNN()
cnn_model.load_weights(data_file + 'weights/Immunogenicity_Predictor/')
after_pca = np.loadtxt(data_file + 'data/DeepImmuno/after_pca.txt') # (21,12)
hla = pd.read_csv(data_file + 'data/DeepImmuno/hla2paratopeTable_aligned.txt',sep='\t')
G = Generator(hidden, seq_len, n_chars, batch_size).to(device)
D = Discriminator(hidden, n_chars, seq_len).to(device)
S = Scorer(hidden, n_chars, seq_len).to(device)
G_optimizer = torch.optim.Adam(G.parameters(), lr=lr,
                               betas=(0.5, 0.9))  # usually should be (0.9,0.999), (momentum,RMSprop)
D_optimizer = torch.optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.9))
S_optimizer = torch.optim.Adam(S.parameters(), lr=lr, betas=(0.5, 0.9))

# train()
data_path_Brain = data_file + 'data/neoepitopes/Brain.4.0_test_mut.csv'
output_dir = data_file + 'result_brain/Goal-directed_WGAN-GP_Gamma0_75/epoch' + str(num_epochs)
data_Brain = pd.read_csv(data_path_Brain)
raw_Brain = data_Brain['peptide'].values
real_dataset_Brain = real_dataset_class(raw_Brain, seq_len, n_chars) # dataset with 6232(# of peptides)*10(peptide length)*21(# of amino acids+'-')

#high immunogenicity data-------------------------------------------------------------------------------------------------
all_data = real_dataset_Brain.process()
all_scores = trainedCNN(after_pca, hla, cnn_model,all_data)  # all_scores.mean() = 0.5953

peptide_data = data_Brain['peptide']

imm_brain = peptide_data
raw_imm_brain = imm_brain.values

real_dataset_imm_brain = real_dataset_class(raw_imm_brain, seq_len, n_chars) # dataset with 6232(# of peptides)*10(peptide length)*21(# of amino acids+'-')

Score_loss = nn.MSELoss()

counter = 0
c_epoch = 0
Threshold = 1
substract_scale = 1
array1, array2, array3, array4, array5, array6, array7, array8, array9, array10 = [], [], [], [], [], [], [], [], [], []
start_train = timeit.default_timer()
for epoch in range(num_epochs):
    start_epoch = timeit.default_timer()
    '''
    The way I understand this trianing process is:
    you first trian the discriminator to minimize the discrepancy between fake and real data, parameters in generator stay constant.
    Then you train the generator, it will adapt to generate more real image.
    It seems like the purpose is just to generate, not discriminate
    '''
    d_fake_losses, d_real_losses, grad_penalties = [], [], []
    S_losses = []
    G_losses, D_losses, W_dist, G_score, G_max_score, G_dvalue, G_unq = [], [], [], [], [], [], []
    real_dataloader = torch.utils.data.DataLoader(real_dataset_imm_brain, batch_size=batch_size, shuffle=True, drop_last=True)
    # https://stackoverflow.com/questions/53570732/get-single-random-example-from-pytorch-dataloader
    for mini_batch in real_dataloader:
        
        real_data = mini_batch.to(device)
        D_optimizer.zero_grad()

        g_count = 0
        while 1:
            fake_data = sample_generator(batch_size)   # generate a mini-batch of fake data 64*10*21      
            if ~torch.isnan(fake_data.sum()):
                break
            g_count = g_count + 1
            print('regenerate-d'+str(g_count))

        d_fake_pred = D(fake_data)    # what's the prediction you get via discriminator
        d_fake_err = d_fake_pred.mean() # different here
        
        d_real_pred = D(real_data)#-substract_scale*mini_batch_score # change here
        d_real_err = d_real_pred.mean()
        
        gradient_penalty = calculate_gradient_penalty(real_data, fake_data)
        if torch.isnan(gradient_penalty):
            import sys
            sys.exit("gradient_penalty! errors!")
        d_error_total = d_fake_err - d_real_err + gradient_penalty
        w_dist =  d_real_err - d_fake_err
        
        
        # https://stackoverflow.com/questions/53975717/pytorch-connection-between-loss-backward-and-optimizer-step
        # Update the parameters
        d_error_total.backward(retain_graph=True)
        D_optimizer.step()      
        
        S_optimizer.zero_grad()
 
        fake_scores = trainedCNN(after_pca, hla, cnn_model,fake_data)
        real_scores = trainedCNN(after_pca, hla, cnn_model,real_data)
        s_fake_pred = S(fake_data)
        s_real_pred = S(real_data)
        s_error_total = Score_loss(s_fake_pred.cuda(),fake_scores.cuda()) + Score_loss(s_real_pred.cuda(),real_scores.cuda()) 
        
        grad_penalties.append(gradient_penalty.detach().cpu().numpy())
        d_real_losses.append(d_real_err.detach().cpu().numpy())
        d_fake_losses.append(d_fake_err.detach().cpu().numpy())
        D_losses.append(d_error_total.detach().cpu().numpy())
        W_dist.append(w_dist.detach().cpu().numpy())
        S_losses.append(s_error_total.detach().cpu().numpy())

        # Update the parameters in the generator every d_steps(10) in mini_batch(64)
        if counter % d_steps == 0:
            s_error_total.backward(retain_graph=True)
            S_optimizer.step() 
            G_optimizer.zero_grad()
            g_count = 0
            unq_count = batch_size
            while 1:
                g_fake_data = sample_generator(batch_size)   # generate a mini-batch of fake data 64*10*21      
                if ~torch.isnan(g_fake_data.sum()):
                    generation = g_fake_data.detach().cpu().numpy()
                    # https://numpy.org/doc/stable/reference/generated/numpy.argmax.html
                    # Returns the indices of the maximum values along an axis
                    hard = np.argmax(generation, axis=2)  # [N,seq_len]
                    # https://stackoverflow.com/questions/45176383/count-occurrences-of-unique-arrays-in-array
                    # count unique generated peptides
                    unq_rows, count = np.unique(hard,axis=0, return_counts=1)
                    unq_count = len(count)
                    break
                g_count = g_count + 1
                print('regenerate-g'+str(g_count))

            dg_fake_pred = D(g_fake_data)
            
            S.eval()
            sg_fake_pred = S(g_fake_data)

            g_fake_scores = trainedCNN(after_pca, hla, cnn_model,g_fake_data) 
            
            D_range = abs(torch.mean(dg_fake_pred)-torch.mean(sg_fake_pred))
            
            score_scale = 1 + D_range
            
            if unq_count == batch_size:
                gamma = 0.75 # <- change here
            else:
                gamma = 0
            unq_rate = unq_count/batch_size
            
            # g_error_total = -torch.mean(dg_fake_pred)-score_scale*torch.mean(sg_fake_pred)
            
            g_error_total = -torch.mean(dg_fake_pred)-gamma*score_scale*torch.mean(sg_fake_pred)
            
            g_error_total.backward()
            G_optimizer.step()
            g_err = g_error_total            
            G_losses.append(g_err.detach().cpu().numpy())
            G_score.append(torch.mean(g_fake_scores).detach().cpu().numpy())
            G_max_score.append(torch.max(g_fake_scores).detach().cpu().numpy())
            G_dvalue.append(torch.mean(dg_fake_pred).detach().cpu().numpy())
            G_unq.append(unq_rate)
            S.train(True)
        else:
            s_error_total.backward()
            S_optimizer.step() 
            
        counter += 1

    summary_string = 'Epoch{0}/{1}: d_real_loss-{2:.2f},d_fake_loss-{3:.2f},d_total_loss-{4:.2f},G_score-{5:.2f},G_total_loss-{6:.2f},W_dist-{7:.2f},S_losses-{8:.2f},G_max_score-{9:.2f},unq_rate-{10:.2f}' \
        .format(epoch + 1, num_epochs, np.mean(d_real_losses), np.mean(d_fake_losses), np.mean(D_losses),np.mean(G_score), 
                np.mean(G_losses), np.mean(W_dist), np.mean(S_losses), np.mean(G_max_score), np.mean(G_unq))
    print(summary_string)
    array1.append(np.mean(d_real_losses))
    array2.append(np.mean(d_fake_losses))
    array3.append(np.mean(D_losses))
    array4.append(np.mean(G_losses))
    array5.append(np.mean(W_dist))
    array6.append(np.mean(G_score))
    array7.append(np.mean(G_dvalue))
    array8.append(np.mean(S_losses))
    array9.append(np.mean(G_max_score))
    array10.append(np.mean(G_unq))

    if epoch % 50 == 49:
        total = []
        for i in range(160):
            generation = sample_generator(batch_size).detach().cpu().numpy()  # [N,seq_len,n_chars] pick the amino acid that has the largest value for one position of the epitope
            #generation = sample_label_generator(batch_size,batch_label,label_size).detach().cpu().numpy()  # [N,seq_len,n_chars] pick the amino acid that has the largest value for one position of the epitope
            hard = np.argmax(generation, axis=2)  # [N,seq_len]
            pseudo = inverse_transform(hard)
            df = pd.DataFrame({'peptide': pseudo, 'HLA': ['HLA-A*0201' for i in range(len(pseudo))],
                               'immunogenicity': [1 for i in range(len(pseudo))]})
            total.append(df)
        df_all = pd.concat(total)
        df_all.to_csv(os.path.join(output_dir,'df_NoDivide_all_epoch{}.csv'.format(i + 1)), index=None)
        torch.save(G.state_dict(), os.path.join(output_dir,'model_epoch_' + str(epoch) + '.pth'))

    c_epoch += 1
    stop_epoch = timeit.default_timer()
    print('Time: '+ str(stop_epoch - start_epoch) + '(s)')
    array8.append(stop_epoch - start_epoch)

stop_train = timeit.default_timer()
print('Training Time: '+ str(stop_train - start_train) + '(s)')
# save the model
torch.save(G.state_dict(), os.path.join(output_dir,'model_epoch_' + str(num_epochs) + '.pth'))


# num_epochs = len(array1)

# start to plot
fig,axes = plt.subplots(nrows=7,ncols=1,figsize=(10,10),gridspec_kw={'hspace':0.4})
ax0 = axes[0]
ax0.plot(np.arange(num_epochs), array1)
ax0.set_ylabel('d_real_losses')

ax1 = axes[1]
ax1.plot(np.arange(num_epochs), array2)
ax1.set_ylabel('d_fake_losses')

ax2 = axes[2]
ax2.plot(np.arange(num_epochs), array3)
ax2.set_ylabel('D_losses')

ax3 = axes[3]
ax3.plot(np.arange(num_epochs), array4)
ax3.set_ylabel('G_losses')

ax4 = axes[4]
ax4.plot(np.arange(num_epochs), array5)
ax4.set_ylabel('W_dist')

ax5 = axes[5]
ax5.plot(np.arange(num_epochs), array6)
ax5.set_ylabel('G_score')

ax6 = axes[6]
ax6.plot(np.arange(num_epochs), array7)
ax6.set_ylabel('G_dvalue')

plt.savefig(os.path.join(output_dir,'diagnose_plot_NoDivide_epoch_' + str(num_epochs) + '.pdf'),bbox_inches='tight')

# test = real_dataset_class(np.array(['YPDTDVILM']),seq_len, n_chars)
# test2 = test.process()
# D(test2)

np.save(output_dir+'/d_real_losses.npy', array1)
np.save(output_dir+'/d_fake_losses.npy', array2)
np.save(output_dir+'/D_losses.npy', array3)
np.save(output_dir+'/G_losses.npy', array4)
np.save(output_dir+'/W_dist.npy', array5)
np.save(output_dir+'/G_score.npy', array6)
np.save(output_dir+'/G_dvalue.npy', array7)
np.save(output_dir+'/runtime.npy', array8)
np.save(output_dir+'/G_max_score.npy', array9)
np.save(output_dir+'/G_unq.npy', array10)

stop_whole = timeit.default_timer()
print('Program RumTime: '+ str(stop_whole - start_whole) + '(s)')

with open(output_dir + 'RunRime.txt', 'w') as f:
    f.write('Training RumTime: '+ str(stop_train - start_train) + '(s)' + '\n')
    f.write('Program RumTime: '+ str(stop_whole - start_whole) + '(s)')