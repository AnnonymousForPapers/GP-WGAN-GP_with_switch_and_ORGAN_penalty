import re
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import os

# Print current directory
print("Current Directory:", os.getcwd())

# Change to your desired directory
os.chdir('D:/PhD thesis/GAN/After TAI/GD_WGAN_GP/Results')

# Confirm the change
print("New Current Directory:", os.getcwd())

# import seaborn as sns
# sns.color_palette("colorblind", 11).as_hex()
colors = [(31,119,180),
          (255,117,14),
          (44,160,44),
          (214,39,40),
          (148,103,189),
          (140,86,75),
          (227,119,194),
          (127,127,127),
          (188,189,34),
          (23,190,207),
          (166,69,0),]

colors = ["#1f77b4",
          "#ff750e",
          "#2ca02c",
          "#d62728",
          "#9467bd",
          "#8c564b",
          "#e377c2",
          "#7f7f7f",
          "#bcbd22",
          "#17becf",
          "#a64500",]

#%% G_score
method_array = ["WGAN-GP",
                "GD_WGAN_wclip_MolGAN_DScalar_Loss_MyPredictor",
                "GD_WGAN_wclip_MolGAN_Gamma0_DScalar_Loss_MyPredictor",
                "GD_WGAN_wclip_OR_MolGAN_Gamma0_DScalar_Loss_MyPredictor",
                "Goal-directed_WGAN-GP_Gamma0_25",
                "Goal-directed_WGAN-GP_Gamma0_5",
                "Goal-directed_WGAN-GP_Gamma0_75",
                "Goal-directed_WGAN-GP",
                "Goal-directed_WGAN-GP_ORGAN",
                # "Goal-directed_WGAN-GP_noRewardNet",
                # "Goal-directed_WGAN-GP_noSscale",
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
              # r"GD-WGAN-GP ($\gamma_{max}=1$) without reward network",
              # r"GD-WGAN-GP ($\gamma_{max}=1$ and $S_{scale}=1$)",
              ]

# Example usage
# avgscore_GD_WGAN_MolGAN_Loss = np.load("GD_WGAN_MolGAN_Loss/epoch1000/G_score.npy")
# avgscore_GD_WGAN_wclip_MolGAN_Loss = np.load("GD_WGAN_wclip_MolGAN_Loss/epoch1000/G_score.npy")
# avgscore_GD_WGAN_wclip_MolGAN_Loss_MyPredictor = np.load("GD_WGAN_wclip_MolGAN_Loss_MyPredictor/epoch1000/G_score.npy")
# avgscore_Goal_directed_WGAN_GP = np.load("Goal-directed_WGAN-GP/epoch1000/G_score.npy")
# avgscore_WGAN_wclip = np.load("WGAN_wclip/epoch1000/G_score.npy")
# avgscore_WGAN_GP = np.load("WGAN-GP/epoch1000/G_score.npy")

epochs = np.arange(1, 1001)

# # Plotting Loss vs. Epoch (Learning curve)

# plt.figure(figsize=(8, 5))
# i = 0
# for method, name in zip(method_array, name_array):
#     avgscore = np.load(method + "/epoch1000/G_score.npy")
#     plt.plot(epochs, avgscore, linestyle='-', color=colors[i], label=name, linewidth=2)
#     i += 1

# plt.xlabel("Epoch", fontsize=20)
# plt.ylabel("Loss", fontsize=20)
# plt.title("Immunogenecity score vs. Epoch from various WGANs", fontsize=20)
# plt.xticks(fontsize=20)
# plt.yticks(fontsize=20)
# plt.grid(True)
# plt.legend(fontsize=16, loc='center', bbox_to_anchor=(0.5, 0., 0, -1.5))
# # plt.tight_layout(rect=[0, 0.1, 1, 1])  # leave space at bottom
# # plt.savefig("wgan_immunogenicity_scores.png", dpi=600)  # You can also use 'pdf' or 'svg'
# plt.show()


import matplotlib.gridspec as gridspec
fig = plt.figure(figsize=(8, 10))  # Taller to accommodate legend
gs = gridspec.GridSpec(nrows=2, ncols=1, height_ratios=[5, 1])  # plot + legend

# Top: Main plot
ax = fig.add_subplot(gs[0])

for i, (method, name) in enumerate(zip(method_array, name_array)):
    avgscore = np.load(method + "/epoch1000/G_score.npy")
    ax.plot(epochs, avgscore, linestyle='-', color=colors[i], label=name, linewidth=2)

ax.set_xlabel("Epoch", fontsize=20)
ax.set_ylabel("Imm. score", fontsize=20)
ax.set_title("Immunogenicity Score vs. Epoch from Various WGANs", fontsize=20)
ax.tick_params(labelsize=16)
ax.grid(True)

# Bottom: Legend
legend_ax = fig.add_subplot(gs[1])
legend_ax.axis("off")  # Hide axes
legend_ax.legend(
    handles=ax.get_legend_handles_labels()[0],
    labels=ax.get_legend_handles_labels()[1],
    loc='center',
    ncol=1,
    fontsize=16,
    frameon=True
)

# Save before showing
plt.tight_layout()
plt.savefig("wgan_immunogenicity_scores.png", dpi=600, bbox_inches="tight")
plt.show()

#%% Unique rate, but missing some data
epochs = np.arange(1, 1001)

import matplotlib.gridspec as gridspec
fig = plt.figure(figsize=(8, 10))  # Taller to accommodate legend
gs = gridspec.GridSpec(nrows=2, ncols=1, height_ratios=[5, 1])  # plot + legend

# Top: Main plot
ax = fig.add_subplot(gs[0])

for i, (method, name) in enumerate(zip(method_array, name_array)):
    avgscore = np.load(method + "/epoch1000/G_unq.npy")
    ax.plot(epochs, avgscore, linestyle='-', color=colors[i], label=name, linewidth=2)

ax.set_xlabel("Epoch", fontsize=20)
ax.set_ylabel("Unique rate", fontsize=20)
ax.set_title("Unique Peptide Rate vs. Epoch from Various WGANs", fontsize=20)
ax.tick_params(labelsize=16)
ax.grid(True)

# Bottom: Legend
legend_ax = fig.add_subplot(gs[1])
legend_ax.axis("off")  # Hide axes
legend_ax.legend(
    handles=ax.get_legend_handles_labels()[0],
    labels=ax.get_legend_handles_labels()[1],
    loc='center',
    ncol=1,
    fontsize=16,
    frameon=True
)

# Save before showing
plt.tight_layout()
plt.savefig("wgan_G_unq_rate.png", dpi=600, bbox_inches="tight")
plt.show()

# Zoomed
fig = plt.figure(figsize=(6, 5))  # Taller to accommodate legend
gs = gridspec.GridSpec(nrows=2, ncols=1, height_ratios=[5, 1])  # plot + legend

# Top: Main plot
ax = fig.add_subplot(gs[0])

for i, (method, name) in enumerate(zip(method_array, name_array)):
    avgscore = np.load(method + "/epoch1000/G_unq.npy")
    ax.plot(epochs, avgscore, linestyle='-', color=colors[i], label=name, linewidth=2)

ax.set_xlabel("Epoch", fontsize=20)
ax.set_ylabel("Unique rate", fontsize=20)
# ax.set_title("Unique Peptide Rate vs. Epoch from Various WGANs", fontsize=20)
ymin, ymax = 0.97, 1.0
ax.set_ylim(ymin, ymax)
ax.tick_params(labelsize=16)
ax.grid(True)

# # Bottom: Legend
# legend_ax = fig.add_subplot(gs[1])
# legend_ax.axis("off")  # Hide axes
# legend_ax.legend(
#     handles=ax.get_legend_handles_labels()[0],
#     labels=ax.get_legend_handles_labels()[1],
#     loc='center',
#     ncol=1,
#     fontsize=16,
#     frameon=True
# )

# Save before showing
plt.tight_layout()
plt.savefig("wgan_G_unq_rate_zoomed.png", dpi=600, bbox_inches="tight")
plt.show()


#%% G and D loss
# # Example usage
# avgGloss_GD_WGAN_MolGAN_Loss = np.load("GD_WGAN_MolGAN_Loss/epoch1000/G_losses.npy")
# avgGloss_GD_WGAN_wclip_MolGAN_Loss = np.load("GD_WGAN_wclip_MolGAN_Loss/epoch1000/G_losses.npy")
# avgGloss_GD_WGAN_wclip_MolGAN_Loss_MyPredictor = np.load("GD_WGAN_wclip_MolGAN_Loss_MyPredictor/epoch1000/G_losses.npy")
# avgGloss_Goal_directed_WGAN_GP = np.load("Goal-directed_WGAN-GP/epoch1000/G_losses.npy")
# avgGloss_WGAN_wclip = np.load("WGAN_wclip/epoch1000/G_losses.npy")
# avgGloss_WGAN_GP = np.load("WGAN-GP/epoch1000/G_losses.npy")

# avgDloss_GD_WGAN_MolGAN_Loss = np.load("GD_WGAN_MolGAN_Loss/epoch1000/D_losses.npy")
# avgDloss_GD_WGAN_wclip_MolGAN_Loss = np.load("GD_WGAN_wclip_MolGAN_Loss/epoch1000/D_losses.npy")
# avgDloss_GD_WGAN_wclip_MolGAN_Loss_MyPredictor = np.load("GD_WGAN_wclip_MolGAN_Loss_MyPredictor/epoch1000/D_losses.npy")
# avgDloss_Goal_directed_WGAN_GP = np.load("Goal-directed_WGAN-GP/epoch1000/D_losses.npy")
# avgDloss_WGAN_wclip = np.load("WGAN_wclip/epoch1000/D_losses.npy")
# avgDloss_WGAN_GP = np.load("WGAN-GP/epoch1000/D_losses.npy")

epochs = np.arange(1, 1001)

# Plotting Loss vs. Epoch (Learning curve)

plt.figure(figsize=(8, 5))

i = 0
for method, name in zip(method_array, name_array):
    avgGloss = np.load(method + "/epoch1000/G_losses.npy")
    avgDloss = np.load(method + "/epoch1000/D_losses.npy")
    plt.plot(epochs, avgGloss, linestyle='-', color=colors[i], label=name + " (Generator loss)", linewidth=3)
    plt.plot(epochs, avgDloss, linestyle=':', color=colors[i], label=name + " (Critic loss)", linewidth=3)
    i += 1
plt.xlabel("Epoch", fontsize=20)
plt.ylabel("Loss", fontsize=20)
plt.title("Loss vs. Epoch from various GAN", fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.grid(True)
plt.legend(fontsize=16, loc='center', bbox_to_anchor=(0.5, 0., 0, -1.5))
plt.show()

#%% Pick best epoch
MolGAN_imm = np.load("GD_WGAN_wclip_MolGAN_DScalar_Loss_MyPredictor" + "/epoch1000/G_score.npy")
MolGAN_Gamma0_imm = np.load("GD_WGAN_wclip_MolGAN_Gamma0_DScalar_Loss_MyPredictor" + "/epoch1000/G_score.npy")
MolGAN_Gamma0_ORGAN_imm = np.load("GD_WGAN_wclip_OR_MolGAN_Gamma0_DScalar_Loss_MyPredictor" + "/epoch1000/G_score.npy")

MolGAN_unq = np.load("GD_WGAN_wclip_MolGAN_DScalar_Loss_MyPredictor" + "/epoch1000/G_unq.npy")
MolGAN_Gamma0_unq = np.load("GD_WGAN_wclip_MolGAN_Gamma0_DScalar_Loss_MyPredictor" + "/epoch1000/G_unq.npy")
MolGAN_Gamma0_ORGAN_unq = np.load("GD_WGAN_wclip_OR_MolGAN_Gamma0_DScalar_Loss_MyPredictor" + "/epoch1000/G_unq.npy")

MolGAN_sum = MolGAN_imm + MolGAN_unq
MolGAN_Gamma0_sum = MolGAN_Gamma0_imm + MolGAN_Gamma0_unq
MolGAN_Gamma0_ORGAN_sum = MolGAN_Gamma0_ORGAN_imm + MolGAN_Gamma0_ORGAN_unq

print(f"Best epoch for MolGAN: {np.argmax(MolGAN_sum) + 1}")
print(f"Best sum for MolGAN: {max(MolGAN_sum)}")

print(f"Best epoch for MolGAN_Gamma0: {np.argmax(MolGAN_Gamma0_sum) + 1}")
print(f"Best sum for MolGAN_Gamma0: {max(MolGAN_Gamma0_sum)}")

print(f"Best epoch for MolGAN_Gamma0_ORGAN: {np.argmax(MolGAN_Gamma0_ORGAN_sum) + 1}")
print(f"Best sum for MolGAN_Gamma0_ORGAN: {max(MolGAN_Gamma0_ORGAN_sum)}")

MolGAN_best_idx = np.argmax(MolGAN_sum)
MolGAN_Gamma0_best_idx = np.argmax(MolGAN_Gamma0_sum)
MolGAN_Gamma0_ORGAN_best_idx = np.argmax(MolGAN_Gamma0_ORGAN_sum)

print(f"Best imm for MolGAN: {MolGAN_imm[MolGAN_best_idx]}")
print(f"Best unq for MolGAN: {MolGAN_unq[MolGAN_best_idx]}")

print(f"Best imm for MolGAN_Gamma0: {MolGAN_Gamma0_imm[MolGAN_Gamma0_best_idx]}")
print(f"Best unq for MolGAN_Gamma0: {MolGAN_Gamma0_unq[MolGAN_Gamma0_best_idx]}")

print(f"Best imm for MolGAN_Gamma0_ORGAN: {MolGAN_Gamma0_ORGAN_imm[MolGAN_Gamma0_ORGAN_best_idx]}")
print(f"Best unq for MolGAN_Gamma0_ORGAN: {MolGAN_Gamma0_ORGAN_unq[MolGAN_Gamma0_ORGAN_best_idx]}")
