The python files are the codes for training the 9 different GANs with brain neoepitopes in the manuscript.

The methods and the corresponding code are shown below:

**WGAN-GP**: WGAN-GP_GPU.py  
**MolGAN ($\lambda_M=0.5$)**: Goal-directed_WGAN_wclip_MolGAN_DScalar_Loss_MyPredictor_GPU.py  
**MolGAN ($\lambda_M=0$)**: Goal-directed_WGAN_wclip_MolGAN_Gamma0_DScalar_Loss_MyPredictor_GPU.py  
**MolGAN ($\lambda_M=0$)** with ORGAN: Goal-directed_WGAN_wclip_OR_MolGAN_Gamma0_DScalar_Loss_MyPredictor_GPU.py  
**GD-WGAN-GP ($\gamma_{max}=0.25$)**: Goal-directed_WGAN-GP_Gamma0_25_GPU.py  
**GD-WGAN-GP ($\gamma_{max}=0.5$)**: Goal-directed_WGAN-GP_Gamma0_5_GPU.py  
**GD-WGAN-GP ($\gamma_{max}=0.75$)**: Goal-directed_WGAN-GP_Gamma0_75_GPU.py  
**GD-WGAN-GP ($\gamma_{max}=1$)**: Goal-directed_WGAN-GP_GPU.py  
**GD_WGAN-GP with ORGAN**: Goal-directed_WGAN-GP_ORGAN_GPU.py

# Folder descriptions
  * The result_brain folder contains the files generated from each python codes, such as the npy files containing the predicted immunogenicity scores during training.
  * The Stats_ep1000 folder contains the codes and the results of the statistics of the models trained after 1000 epochs.
  * The Stats_MolGAN_Best folder contains the codes and the results of the statistics of the models trained trained using the MolGAN's methods and selected by the largest sum of imm. score and unique rate.
  * The Training_log folder contains the outputs generated from each python codes.
