'''
Revised from https://github.com/frankligy/DeepImmuno
DeepImmuno: deep learning-empowered prediction and generation of 
immunogenic peptides for T-cell immunity, Briefings in Bioinformatics, 
May 03 2021 (https://doi.org/10.1093/bib/bbab160)

Diffusion replacement based on:
Austin et al., Structured Denoising Diffusion Models in Discrete State-Spaces,
NeurIPS 2021 (D3PM), https://proceedings.neurips.cc/paper/2021/hash/958c530554f78bcd8e97125b70e6973d-Abstract.html
Official implementation: https://github.com/google-research/google-research/tree/master/d3pm

The D3PM paper's text experiments are UNCONDITIONAL, not goal/property-directed.
Therefore the Transformer + D3PM process follows Austin et al., while the
immunogenicity objective remains the goal-directed Scorer mechanism inherited
from the original ORGAN/DeepImmuno code.
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
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--num_epochs", type=int, default=1000)
parser.add_argument("--seed", type=int, default=53)
args = parser.parse_args()

seed = args.seed
num_epochs = args.num_epochs

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

print(f"Random seed: {seed}")
print(f"Number of epochs: {num_epochs}")

def count_params(model, trainable_only=False):
    params = (p for p in model.parameters() if (p.requires_grad or not trainable_only))
    return sum(p.numel() for p in params)

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

def build_rotary_embeddings(seq_len, head_dim, base=10000):
    """RoPE frequencies, matched to the attached Transformer generator."""
    pos = torch.arange(seq_len, dtype=torch.float32)
    freqs = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
    angles = torch.outer(pos, freqs)
    return torch.cos(angles), torch.sin(angles)


def apply_rotary(x, cos, sin):
    x1, x2 = x[..., ::2], x[..., 1::2]
    x_rot = torch.stack(
        [x1 * cos - x2 * sin, x1 * sin + x2 * cos],
        dim=-1,
    )
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
    """Unmasked grouped-query attention, matching the attached architecture."""
    def __init__(self, embed_dim, q_heads, kv_heads):
        super().__init__()
        assert embed_dim % q_heads == 0
        assert q_heads % kv_heads == 0
        self.q_heads = q_heads
        self.kv_heads = kv_heads
        self.head_dim = embed_dim // q_heads
        assert self.head_dim % 2 == 0, 'RoPE requires an even head dimension.'
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

        # No causal mask: every peptide position can attend to every position.
        att = (q @ k.transpose(-2, -1)) / self.scale
        weights = F.softmax(att, dim=-1)
        out = weights @ v
        return self.out_proj(out.transpose(1, 2).reshape(B, T, -1))


class FFNLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(dim, int(4 * dim)),
            nn.GELU(),
            nn.Linear(int(4 * dim), dim),
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
        cos = cos.to(x.device).unsqueeze(0).unsqueeze(0)
        sin = sin.to(x.device).unsqueeze(0).unsqueeze(0)

        x = self.attn(x, cos, sin) + x
        x = self.rms1(x)
        x = self.ffn(x) + x
        x = self.rms2(x)
        return x


class Generator(nn.Module):
    """
    D3PM x0-prediction denoiser using the same Transformer style as the
    attached Goal-directed_WGAN-GP_TransformerNoMaskL2 generator:
      - hidden/embed dim 128
      - 2 Transformer layers
      - 8 query heads / 8 KV heads
      - RoPE
      - RMSNorm
      - 4x GELU FFN
      - no causal attention mask

    The GAN noise-input interface is replaced only as required by D3PM:
      noisy categorical peptide x_t + diffusion timestep t -> x0 logits.
    """
    def __init__(self, hidden, seq_len, n_chars, batch_size, num_diffusion_steps,
                 num_layers=2, q_heads=8, kv_heads=8):
        super().__init__()
        self.seq_len = seq_len
        self.n_chars = n_chars
        self.batch_size = batch_size
        self.hidden = hidden

        # D3PM categorical input projection instead of GAN fc1(noise).
        self.input_proj = nn.Linear(n_chars, hidden)

        # D3PM needs explicit diffusion-time conditioning.
        self.time_embedding = nn.Embedding(num_diffusion_steps, hidden)

        self.layers = nn.ModuleList([
            TransformerBlock(hidden, q_heads, kv_heads)
            for _ in range(num_layers)
        ])
        self.proj = nn.Linear(hidden, n_chars)

    def forward(self, noisy_data, t):
        # noisy_data: [B, L, K] one-hot x_t
        # t:          [B] zero-based diffusion timestep
        h = self.input_proj(noisy_data)
        h = h + self.time_embedding(t).unsqueeze(1)
        for layer in self.layers:
            h = layer(h)
        return self.proj(h)  # [B, L, K] x0 logits


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
                pep = pep[0:5] + '-' + pep[5:]
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


# D3PM-uniform diffusion process
# Austin et al., NeurIPS 2021:
#   Eq. (2): categorical forward transition q(x_t | x_{t-1})
#   Eq. (3): t-step marginal and exact posterior
#   Eq. (4): x_0-parameterized learned reverse process
#   Eq. (5): hybrid variational-bound + auxiliary x_0 cross-entropy loss
#
# The uniform transition matrix below is the paper's D3PM-uniform special case:
#   Q_t = (1 - beta_t) I + (beta_t / K) 11^T
# and the cosine beta schedule follows the authors' official D3PM implementation.
class D3PMUniform:
    def __init__(self, num_steps, n_chars, device):
        self.num_steps = num_steps
        self.n_chars = n_chars
        self.device = device
        self.eps = 1e-8

        # Official D3PM cosine schedule for uniform diffusion.
        # google-research/d3pm/images/diffusion_categorical.py
        steps = torch.arange(num_steps + 1, dtype=torch.float64, device=device) / num_steps
        alpha_bar = torch.cos((steps + 0.008) / 1.008 * np.pi / 2)
        betas = 1.0 - alpha_bar[1:] / alpha_bar[:-1]
        betas = torch.clamp(betas, max=0.999)
        self.betas = betas.float()

        # D3PM-uniform transition matrices Q_t.
        eye = torch.eye(n_chars, dtype=torch.float64, device=device)
        uniform = torch.ones((n_chars, n_chars), dtype=torch.float64, device=device) / n_chars
        q_one_step = []
        for beta in betas:
            q_one_step.append((1.0 - beta) * eye + beta * uniform)
        self.Q = torch.stack(q_one_step, dim=0).float()  # [T,K,K]

        # Qbar_t = Q_1 Q_2 ... Q_t (paper Eq. 3).
        qbar = []
        running = torch.eye(n_chars, dtype=torch.float64, device=device)
        for q in q_one_step:
            running = running @ q
            qbar.append(running.clone())
        self.Qbar = torch.stack(qbar, dim=0).float()     # [T,K,K]

        # For posterior calculations: I when t=0, otherwise Qbar_{t-1}.
        self.Qbar_prev = torch.cat(
            [torch.eye(n_chars, device=device).unsqueeze(0), self.Qbar[:-1]],
            dim=0,
        )

    def q_sample(self, x_start_idx, t):
        """Sample x_t ~ q(x_t | x_0), corresponding to D3PM Eq. (3)."""
        x_start = F.one_hot(x_start_idx, num_classes=self.n_chars).float()  # [B,L,K]
        qbar_t = self.Qbar[t]                                               # [B,K,K]
        probs = torch.einsum('blk,bkj->blj', x_start, qbar_t)               # [B,L,K]
        x_t = torch.multinomial(probs.reshape(-1, self.n_chars), 1)
        return x_t.view(x_start_idx.shape)

    def posterior_from_start_probs(self, start_probs, x_t_idx, t):
        """
        Return the categorical distribution over x_{t-1}.

        For true one-hot x_0 this is q(x_{t-1}|x_t,x_0), paper Eq. (3).
        For predicted p_tilde(x_0|x_t), this gives the D3PM x_0
        parameterization of p_theta(x_{t-1}|x_t), paper Eq. (4).
        """
        B, L, K = start_probs.shape
        out = torch.empty_like(start_probs)

        # In the authors' implementation, t=0 is the first noisy state and
        # reversing it returns x_start directly.
        zero_mask = (t == 0)
        if zero_mask.any():
            out[zero_mask] = start_probs[zero_mask]

        nz_mask = ~zero_mask
        if nz_mask.any():
            probs0 = start_probs[nz_mask]       # [b,L,K]
            xt = x_t_idx[nz_mask]               # [b,L]
            tnz = t[nz_mask]                    # [b]

            Qt = self.Q[tnz]                    # [b,K,K]
            Qbar_prev = self.Qbar_prev[tnz]     # [b,K,K]

            # fact1_j = q(x_t | x_{t-1}=j) = Q_t[j, x_t]
            xt_expand = xt.unsqueeze(1).expand(-1, K, -1)                  # [b,K,L]
            fact1 = torch.gather(Qt, 2, xt_expand).transpose(1,2)           # [b,L,K]

            # fact2_j = sum_x0 p(x0) q(x_{t-1}=j | x0)
            fact2 = torch.einsum('blk,bkj->blj', probs0, Qbar_prev)         # [b,L,K]

            posterior = fact1 * fact2
            posterior = posterior / posterior.sum(dim=-1, keepdim=True).clamp_min(self.eps)
            out[nz_mask] = posterior

        return out

    def training_loss(self, model, x_start_onehot, hybrid_coeff):
        """
        D3PM hybrid loss from Austin et al. Eq. (5):
            L = L_vb + lambda * CE(x_0, p_tilde(x_0|x_t)).

        The randomly selected timestep and the t=0 decoder-NLL handling follow
        the authors' official implementation.
        """
        B = x_start_onehot.shape[0]
        x_start_idx = torch.argmax(x_start_onehot, dim=-1)                  # [B,L]
        t = torch.randint(0, self.num_steps, (B,), device=self.device)      # [B]
        x_t_idx = self.q_sample(x_start_idx, t)                              # [B,L]
        x_t_onehot = F.one_hot(x_t_idx, num_classes=self.n_chars).float()   # [B,L,K]

        pred_x0_logits = model(x_t_onehot, t)                                # [B,L,K]
        pred_x0_probs = torch.softmax(pred_x0_logits, dim=-1)

        true_post = self.posterior_from_start_probs(x_start_onehot, x_t_idx, t)
        model_post = self.posterior_from_start_probs(pred_x0_probs, x_t_idx, t)

        # Variational-bound term: decoder NLL at the first step, KL otherwise.
        log_true = torch.log(true_post.clamp_min(self.eps))
        log_model = torch.log(model_post.clamp_min(self.eps))
        kl = torch.sum(true_post * (log_true - log_model), dim=-1).mean(dim=-1) / np.log(2.0)

        p_x0 = model_post.gather(-1, x_start_idx.unsqueeze(-1)).squeeze(-1)
        decoder_nll = -torch.log(p_x0.clamp_min(self.eps)).mean(dim=-1) / np.log(2.0)
        vb_loss = torch.where(t == 0, decoder_nll, kl)

        ce = F.cross_entropy(
            pred_x0_logits.reshape(-1, self.n_chars),
            x_start_idx.reshape(-1),
            reduction='none',
        ).view(B, -1).mean(dim=-1) / np.log(2.0)

        hybrid_loss = vb_loss + hybrid_coeff * ce
        return hybrid_loss, vb_loss, ce, pred_x0_probs, pred_x0_logits, t

    @torch.no_grad()
    def sample(self, model, batch_size, seq_len):
        """Ancestral D3PM sampling, starting from the uniform stationary distribution."""
        x = torch.randint(0, self.n_chars, (batch_size, seq_len), device=self.device)

        for step in range(self.num_steps - 1, -1, -1):
            t = torch.full((batch_size,), step, device=self.device, dtype=torch.long)
            x_onehot = F.one_hot(x, num_classes=self.n_chars).float()
            pred_x0_logits = model(x_onehot, t)
            pred_x0_probs = torch.softmax(pred_x0_logits, dim=-1)
            reverse_probs = self.posterior_from_start_probs(pred_x0_probs, x, t)

            if step == 0:
                # Same no-noise final step convention as the official D3PM code.
                x = torch.argmax(reverse_probs, dim=-1)
            else:
                x = torch.multinomial(reverse_probs.reshape(-1, self.n_chars), 1)
                x = x.view(batch_size, seq_len)

        return F.one_hot(x, num_classes=self.n_chars).float()


def sample_generator(batch_size):
    # Preserve the original helper name so the CSV-generation block needs
    # almost no structural change.
    return diffusion.sample(G, batch_size, seq_len)


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


# ============================================================================
# Normal single-run experiment
# Denoiser architecture matched to attached TransformerNoMaskL2 style.
# Train one D3PM with T=1000, batch size 64, for 1000 epochs.
# At the end of training, generate exactly 10,000 sequences using the full
# stochastic reverse diffusion sampler and report exact-sequence unique rate.
# ============================================================================

batch_size = 64
lr = 0.0001
# num_epochs is controlled by --num_epochs
seq_len = 10
hidden = 128
n_chars = 21

transformer_layers = 2
transformer_q_heads = 8
transformer_kv_heads = 8
hybrid_coeff = 0.0
score_guidance_weight = 1.0

num_diffusion_steps = 1000
NUM_FINAL_SAMPLES = 10000
BASE_SEED = seed

MODE = "ORGAN_Switching_Gamma0_25"
MODE_FOLDER = "D3PM_ORGAN_Switching_Gamma0_25_AttachedTransformer"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')

data_file = '../'

# Load fixed DeepImmuno CNN and training data.
cnn_model = seperateCNN()
cnn_model.load_weights(data_file + 'weights/Immunogenicity_Predictor/')
after_pca = np.loadtxt(data_file + 'data/DeepImmuno/after_pca.txt')
hla = pd.read_csv(data_file + 'data/DeepImmuno/hla2paratopeTable_aligned.txt', sep='\t')

data_path_Brain = data_file + 'data/neoepitopes/Brain.4.0_test_mut.csv'
data_Brain = pd.read_csv(data_path_Brain)
raw_Brain = data_Brain['peptide'].values
real_dataset_imm_brain = real_dataset_class(raw_Brain, seq_len, n_chars)

Score_loss = nn.MSELoss()

output_root = os.path.join(
    data_file,
    'result_brain',
    MODE_FOLDER + '_seed' + str(seed),
    'epoch' + str(num_epochs),
)
os.makedirs(output_root, exist_ok=True)


def reset_seed(seed):
    """Use the same initialization/data-order seed for every T experiment."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def generate_exactly_n(diffusion, G, n_samples, batch_size, seq_len):
    """Generate exactly n_samples with the full stochastic reverse sampler."""
    peptides = []
    G.eval()
    with torch.no_grad():
        while len(peptides) < n_samples:
            current_bs = min(batch_size, n_samples - len(peptides))
            generation = diffusion.sample(G, current_bs, seq_len).detach().cpu().numpy()
            hard = np.argmax(generation, axis=2)
            peptides.extend(inverse_transform(hard))
    G.train(True)
    return peptides


def run_training():
    print('\n' + '=' * 90)
    print(
        f'MODE={MODE} | SEED={seed} | T={num_diffusion_steps} | B={batch_size} | '
        f'EPOCHS={num_epochs} | FINAL_SAMPLES={NUM_FINAL_SAMPLES}'
    )
    print('=' * 90)

    reset_seed(BASE_SEED)

    output_dir = output_root
    os.makedirs(output_dir, exist_ok=True)

    G = Generator(
        hidden, seq_len, n_chars, batch_size, num_diffusion_steps,
        num_layers=transformer_layers,
        q_heads=transformer_q_heads,
        kv_heads=transformer_kv_heads,
    ).to(device)
    S = Scorer(hidden, n_chars, seq_len).to(device)
    diffusion = D3PMUniform(num_diffusion_steps, n_chars, device)

    print(f'Total diffusion-model params: {count_params(G):,}')

    G_optimizer = torch.optim.Adam(G.parameters(), lr=lr)
    S_optimizer = torch.optim.Adam(S.parameters(), lr=lr, betas=(0.5, 0.9))

    array_vb, array_ce, array_diffusion, array_total = [], [], [], []
    array_goal_score, array_penalized_goal_score = [], []
    array_cnn_score, array_s_loss, array_max_score, array_unq = [], [], [], []
    array_runtime = []

    best_checkpoint_score = -float("inf")
    best_checkpoint_epoch = None
    best_checkpoint_imm_score = None
    best_checkpoint_unique_ratio = None

    start_train = timeit.default_timer()

    # Dedicated DataLoader RNG for reproducible shuffling.
    loader_generator = torch.Generator()
    loader_generator.manual_seed(BASE_SEED)

    for epoch in range(num_epochs):
        start_epoch = timeit.default_timer()

        vb_losses, ce_losses, diffusion_losses, total_losses = [], [], [], []
        goal_scores, penalized_goal_scores, cnn_scores = [], [], []
        max_scores, unq_rates, S_losses = [], [], []

        real_dataloader = torch.utils.data.DataLoader(
            real_dataset_imm_brain,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            generator=loader_generator,
        )

        for mini_batch in real_dataloader:
            real_data = mini_batch.to(device)

            # 1) D3PM training loss: one randomly sampled timestep per example.
            G_optimizer.zero_grad()
            hybrid_loss_b, vb_loss_b, ce_loss_b, pred_x0_probs, pred_x0_logits, t = \
                diffusion.training_loss(G, real_data, hybrid_coeff)

            # 2) Retain original differentiable scorer fitting for matched diagnostics.
            S_optimizer.zero_grad()
            fake_scores = trainedCNN(after_pca, hla, cnn_model, pred_x0_probs.detach())
            real_scores = trainedCNN(after_pca, hla, cnn_model, real_data)

            s_fake_pred = S(pred_x0_probs.detach())
            s_real_pred = S(real_data)
            s_error_total = (
                Score_loss(s_fake_pred, fake_scores.to(device))
                + Score_loss(s_real_pred, real_scores.to(device))
            )
            s_error_total.backward()
            S_optimizer.step()

            # 3) Reward calculation.  Only the total-loss rule differs by MODE.
            S.eval()
            for p in S.parameters():
                p.requires_grad_(False)

            hard_for_counts = torch.argmax(pred_x0_probs.detach(), dim=2)
            unique_rows = torch.unique(hard_for_counts, dim=0)
            unq_count = unique_rows.shape[0]
            unq_rate = unq_count / pred_x0_probs.shape[0]

            sample_goal_scores = S(pred_x0_probs)
            goal_score = torch.mean(sample_goal_scores)

            # ORGAN switching method:
            # apply the goal/reward term only when every peptide in the
            # generated minibatch is unique; otherwise switch it off.
            if unq_count == pred_x0_probs.shape[0]:
                gamma = 0.25
            else:
                gamma = 0.0

            penalized_goal_score = gamma * goal_score

            diffusion_loss = torch.mean(hybrid_loss_b)
            total_loss = diffusion_loss - score_guidance_weight * penalized_goal_score

            total_loss.backward()
            G_optimizer.step()

            for p in S.parameters():
                p.requires_grad_(True)
            S.train(True)

            vb_losses.append(torch.mean(vb_loss_b).detach().cpu().item())
            ce_losses.append(torch.mean(ce_loss_b).detach().cpu().item())
            diffusion_losses.append(diffusion_loss.detach().cpu().item())
            total_losses.append(total_loss.detach().cpu().item())
            goal_scores.append(goal_score.detach().cpu().item())
            penalized_goal_scores.append(penalized_goal_score.detach().cpu().item())
            cnn_scores.append(torch.mean(fake_scores).detach().cpu().item())
            max_scores.append(torch.max(fake_scores).detach().cpu().item())
            unq_rates.append(unq_rate)
            S_losses.append(s_error_total.detach().cpu().item())

        epoch_runtime = timeit.default_timer() - start_epoch
        array_runtime.append(epoch_runtime)

        summary_string = (
            'T{0} Epoch{1}/{2}: vb_loss-{3:.4f},ce_loss-{4:.4f},diffusion_loss-{5:.4f},'
            'goal_score-{6:.4f},penalized_goal_score-{7:.4f},total_loss-{8:.4f},'
            'CNN_score-{9:.4f},S_loss-{10:.4f},max_score-{11:.4f},unq_rate-{12:.4f},time-{13:.2f}s'
        ).format(
            num_diffusion_steps,
            epoch + 1,
            num_epochs,
            np.mean(vb_losses),
            np.mean(ce_losses),
            np.mean(diffusion_losses),
            np.mean(goal_scores),
            np.mean(penalized_goal_scores),
            np.mean(total_losses),
            np.mean(cnn_scores),
            np.mean(S_losses),
            np.mean(max_scores),
            np.mean(unq_rates),
            epoch_runtime,
        )
        print(summary_string)

        array_vb.append(np.mean(vb_losses))
        array_ce.append(np.mean(ce_losses))
        array_diffusion.append(np.mean(diffusion_losses))
        array_total.append(np.mean(total_losses))
        array_goal_score.append(np.mean(goal_scores))
        array_penalized_goal_score.append(np.mean(penalized_goal_scores))
        array_cnn_score.append(np.mean(cnn_scores))
        array_s_loss.append(np.mean(S_losses))
        array_max_score.append(np.mean(max_scores))
        array_unq.append(np.mean(unq_rates))

        # Evaluate a fresh full reverse-diffusion batch every 50 epochs.
        if (epoch + 1) % 50 == 0:
            G.eval()
            with torch.no_grad():
                checkpoint_fake_data = diffusion.sample(
                    G, batch_size, seq_len
                )

            checkpoint_hard = torch.argmax(
                checkpoint_fake_data, dim=2
            ).detach().cpu().numpy()
            checkpoint_unique_count = len(
                np.unique(checkpoint_hard, axis=0)
            )
            checkpoint_unique_ratio = (
                checkpoint_unique_count / float(batch_size)
            )

            checkpoint_imm_scores = trainedCNN(
                after_pca, hla, cnn_model, checkpoint_fake_data
            )
            checkpoint_mean_imm = float(
                torch.mean(checkpoint_imm_scores).detach().cpu().item()
            )

            checkpoint_score = (
                checkpoint_mean_imm + checkpoint_unique_ratio
            )

            print(
                "Checkpoint evaluation: "
                f"epoch={epoch + 1}, "
                f"mean_imm_score={checkpoint_mean_imm:.6f}, "
                f"unique={checkpoint_unique_count}/{batch_size}, "
                f"unique_ratio={checkpoint_unique_ratio:.6f}, "
                f"sum={checkpoint_score:.6f}"
            )

            if checkpoint_score > best_checkpoint_score:
                best_checkpoint_score = checkpoint_score
                best_checkpoint_epoch = epoch + 1
                best_checkpoint_imm_score = checkpoint_mean_imm
                best_checkpoint_unique_ratio = checkpoint_unique_ratio

                torch.save(
                    G.state_dict(),
                    os.path.join(output_dir, "model_best.pth")
                )

                with open(
                    os.path.join(output_dir, "best_model_info.txt"), "w"
                ) as f:
                    f.write(f"seed: {seed}\n")
                    f.write(f"num_epochs_requested: {num_epochs}\n")
                    f.write(f"best_epoch: {best_checkpoint_epoch}\n")
                    f.write(
                        f"mean_immunogenicity_score: "
                        f"{best_checkpoint_imm_score:.10f}\n"
                    )
                    f.write(
                        f"unique_ratio: "
                        f"{best_checkpoint_unique_ratio:.10f}\n"
                    )
                    f.write(
                        f"checkpoint_score_sum: "
                        f"{best_checkpoint_score:.10f}\n"
                    )

                print(
                    f"[BEST MODEL UPDATED] epoch={best_checkpoint_epoch}, "
                    f"sum={best_checkpoint_score:.6f}"
                )

            G.train(True)

    train_seconds = timeit.default_timer() - start_train

    # Save final checkpoint.
    torch.save(
        G.state_dict(),
        os.path.join(output_dir, "model_last.pth")
    )

    # For short runs (<50 epochs), also define the last checkpoint as best.
    if best_checkpoint_epoch is None:
        G.eval()
        with torch.no_grad():
            checkpoint_fake_data = diffusion.sample(
                G, batch_size, seq_len
            )

        checkpoint_hard = torch.argmax(
            checkpoint_fake_data, dim=2
        ).detach().cpu().numpy()
        checkpoint_unique_count = len(
            np.unique(checkpoint_hard, axis=0)
        )
        best_checkpoint_unique_ratio = (
            checkpoint_unique_count / float(batch_size)
        )

        checkpoint_imm_scores = trainedCNN(
            after_pca, hla, cnn_model, checkpoint_fake_data
        )
        best_checkpoint_imm_score = float(
            torch.mean(checkpoint_imm_scores).detach().cpu().item()
        )
        best_checkpoint_score = (
            best_checkpoint_imm_score + best_checkpoint_unique_ratio
        )
        best_checkpoint_epoch = num_epochs

        torch.save(
            G.state_dict(),
            os.path.join(output_dir, "model_best.pth")
        )

        with open(
            os.path.join(output_dir, "best_model_info.txt"), "w"
        ) as f:
            f.write(f"seed: {seed}\n")
            f.write(f"num_epochs_requested: {num_epochs}\n")
            f.write(f"best_epoch: {best_checkpoint_epoch}\n")
            f.write(
                f"mean_immunogenicity_score: "
                f"{best_checkpoint_imm_score:.10f}\n"
            )
            f.write(
                f"unique_ratio: "
                f"{best_checkpoint_unique_ratio:.10f}\n"
            )
            f.write(
                f"checkpoint_score_sum: "
                f"{best_checkpoint_score:.10f}\n"
            )

        G.train(True)

    print(
        f"Best checkpoint: epoch={best_checkpoint_epoch}, "
        f"mean_imm_score={best_checkpoint_imm_score:.6f}, "
        f"unique_ratio={best_checkpoint_unique_ratio:.6f}, "
        f"sum={best_checkpoint_score:.6f}"
    )
    print(f"Last checkpoint saved at epoch {num_epochs}")

    # ------------------------------------------------------------------------
    # Final evaluation: exactly 10,000 samples from the COMPLETE stochastic
    # reverse diffusion chain, not argmax(pred_x0_probs) from training.
    # ------------------------------------------------------------------------
    start_sample = timeit.default_timer()
    final_peptides = generate_exactly_n(
        diffusion, G, NUM_FINAL_SAMPLES, batch_size, seq_len
    )
    sampling_seconds = timeit.default_timer() - start_sample

    final_unique_count = len(set(final_peptides))
    final_unique_rate = final_unique_count / len(final_peptides)

    print(
        f'FINAL FULL-SAMPLER RESULT | T={num_diffusion_steps} | '
        f'unique={final_unique_count}/{NUM_FINAL_SAMPLES} | '
        f'unique_rate={final_unique_rate:.6f} | '
        f'sampling_time={sampling_seconds:.2f}s'
    )

    counts = pd.Series(final_peptides).value_counts()
    generated_df = pd.DataFrame({'peptide': final_peptides})
    generated_df['repetition_count'] = generated_df['peptide'].map(counts)
    generated_df.to_csv(
        os.path.join(output_dir, 'generated_10000_full_reverse_sampler.csv'),
        index=False,
    )

    # Save training diagnostics.
    np.save(os.path.join(output_dir, 'VB_losses.npy'), array_vb)
    np.save(os.path.join(output_dir, 'CE_losses.npy'), array_ce)
    np.save(os.path.join(output_dir, 'diffusion_losses.npy'), array_diffusion)
    np.save(os.path.join(output_dir, 'total_losses.npy'), array_total)
    np.save(os.path.join(output_dir, 'goal_score.npy'), array_goal_score)
    np.save(os.path.join(output_dir, 'penalized_goal_score.npy'), array_penalized_goal_score)
    np.save(os.path.join(output_dir, 'CNN_score.npy'), array_cnn_score)
    np.save(os.path.join(output_dir, 'S_losses.npy'), array_s_loss)
    np.save(os.path.join(output_dir, 'max_score.npy'), array_max_score)
    np.save(os.path.join(output_dir, 'training_argmax_unq_rate.npy'), array_unq)
    np.save(os.path.join(output_dir, 'runtime.npy'), array_runtime)

    with open(os.path.join(output_dir, 'RunTime.txt'), 'w') as f:
        f.write(f'Seed: {seed}\n')
        f.write(f'Number of epochs: {num_epochs}\n')
        f.write(f'Training Time: {train_seconds}(s)\n')
        f.write(f'Final 10000 Sampling Time: {sampling_seconds}(s)\n')

    return {
        'mode': MODE,
        'seed': seed,
        'T': num_diffusion_steps,
        'batch_size': batch_size,
        'epochs': num_epochs,
        'num_generated': NUM_FINAL_SAMPLES,
        'num_unique': final_unique_count,
        'unique_rate': final_unique_rate,
        'training_time_s': train_seconds,
        'sampling_time_s': sampling_seconds,
        'best_epoch': best_checkpoint_epoch,
        'best_mean_imm_score': best_checkpoint_imm_score,
        'best_unique_ratio': best_checkpoint_unique_ratio,
        'best_checkpoint_score_sum': best_checkpoint_score,
    }


result = run_training()
summary_df = pd.DataFrame([result])
summary_path = os.path.join(output_root, 'final_unique_rate_summary.csv')
summary_df.to_csv(summary_path, index=False)

print('\n' + '=' * 90)
print('FINAL RUN SUMMARY')
print('=' * 90)
print(summary_df.to_string(index=False))
print(f'\nSaved summary to: {summary_path}')
