# GD-WGAN-GP for Peptide Generation

This repository contains the code and data corresponding to the revised manuscript.

## Code version

This repository corresponds to version 1.0 of the code used for the revised manuscript. The manuscript release is identified by Git tag `v1.0`.

## Environments

Two Conda environment files are provided:

- `environment_tf.yml`: environment used for the main GD-WGAN-GP, WGAN-GP, MolGAN, D3PM, DeepImmuno, and evaluation experiments (Python 3.11.5).
- `environment_pepinvent.yml`: environment used for the PepINVENT baseline (Python 3.11.16).

The environments can be created using:

```bash
conda env create -f environment_tf.yml
conda env create -f environment_pepinvent.yml
```

## Training datasets

The final training datasets used in the manuscript are provided in `data/neoepitopes/`:

- `Bladder.4.0_test_mut.csv`: 6,232 bladder cancer-associated neoantigen sequences.
- `Brain.4.0_test_mut.csv`: 2,454 brain cancer-associated neoantigen sequences.

The datasets contain 9- and 10-mer sequences selected for HLA-A*02:01 with a predicted binding affinity <500 nM using NetMHCpan 4.0. The HLA allele is stored as `HLA-A*0201` in the CSV files, corresponding to HLA-A*02:01.

Additional information about the source of the datasets is provided in `data/neoepitopes/Readme.txt`.

## Random seeds

The main experiments reported across multiple random seeds use seeds 0 through 10. The seed is passed to each training script using the `--seed` argument.

For the PyTorch-based models, the specified seed is applied to Python's `random` module, NumPy, PyTorch, and CUDA when available. The corresponding seed is also used during peptide generation and evaluation.

Supplementary experiments reported using a single random seed use seed 0 unless otherwise specified.

## Peptide preprocessing

The training datasets contain peptide sequences of length 9 or 10.

Because the generator represents all peptides using a fixed sequence length of 10, a 9-mer peptide is padded by inserting the placeholder character `-` after the fifth amino acid:

```text
ABCDEFGHI -> ABCDE-FGHI
```

A 10-mer peptide is used without padding.

The same padding convention is used consistently for the bladder and brain datasets. When generated sequences are converted back to peptide sequences for evaluation, the placeholder character is removed so that the resulting peptides have their natural 9- or 10-amino-acid lengths.


## Checkpoint selection

Training is performed for 1,000 epochs unless otherwise specified. For models with best-checkpoint selection, checkpoint evaluation is performed every 50 epochs using 64 generated peptides.

The checkpoint score is

\[
\text{checkpoint score}
=
\text{mean predicted immunogenicity}
+
\text{unique peptide ratio}.
\]

The checkpoint with the highest score is saved as `model_best.pth`, and the generator after the final training epoch is saved as `model_last.pth`. PepINVENT checkpoints use the corresponding `.chkpt` extension.

The checkpoint used for each reported experiment is specified by the corresponding reproduction or configuration script. Unless otherwise specified in the manuscript, reported results use the final checkpoint; results using the best checkpoint are explicitly identified.

## Evaluation procedure

Unless otherwise specified, 10,000 peptide sequences are generated from each trained generator for evaluation. The random seed used for generation corresponds to the seed of the evaluated model.

Generated sequences are converted to peptide sequences, filtered to retain valid 9- and 10-mer peptides, and duplicates are removed. Predictor-based evaluations are then performed on the resulting unique 9- and 10-mer peptides.

The unified evaluation pipeline is implemented in:

`unified_peptide_evaluation/run_all_evaluations.py`

The evaluation pipeline supports the following predictors and metrics used in the manuscript:

- DeepImmuno-CNN predicted immunogenicity.
- IEDB Class I immunogenicity prediction.
- NetMHCpan 4.0 predicted binding affinity.
- NetMHCpan 4.1 predicted binding affinity.
- PEPMatch comparison against the human proteome.
- PepSySco predicted peptide synthesis score.
- Exact-match comparison with the TCGA-BLCA-derived peptide dataset.
- Peptide diversity and novelty metrics, including edit-distance- and nearest-neighbor-based measures.

For NetMHCpan evaluation, peptides with predicted binding affinity below 500 nM are considered predicted binders, as described in the manuscript.

The evaluation script can select `model_last.pth`, `model_best.pth`, or an explicitly specified checkpoint using the `--checkpoint` argument. Unless otherwise specified in the manuscript, reported results use `model_last.pth`.

The evaluation code used for the manuscript is provided under `unified_peptide_evaluation/`.

## Reproducing the experiments

Three scripts are provided for reproducing the training and evaluation workflows.

### Reproducing a single experiment

`reproduce_experiment.sh` trains one model for a specified random seed and performs the evaluation pipeline.

For example:

```bash
./reproduce_experiment.sh gd_gamma1 0
```

The supported model names are:

- `wgan`: WGAN-GP
- `molgan050`: MolGAN (\(\lambda_M=0.5\))
- `molgan0`: MolGAN (\(\lambda_M=0\))
- `molgan0_organ`: MolGAN (\(\lambda_M=0\)) with the revised ORGAN-based repetition penalty
- `gd_gamma025`: GD-WGAN-GP (\(\gamma_{\max}=0.25\))
- `gd_gamma1`: GD-WGAN-GP (\(\gamma_{\max}=1\))
- `gd_organ`: GD-WGAN-GP with the revised ORGAN-based repetition penalty

By default, training is performed for 1,000 epochs, 10,000 peptide sequences are generated for evaluation, and `model_last.pth` is evaluated. These settings can be changed using the `EPOCHS`, `NUM_SAMPLES`, and `CHECKPOINT` environment variables.

For example, an existing training run can be evaluated using the best checkpoint without repeating training:

```bash
SKIP_TRAIN=1 CHECKPOINT=best ./reproduce_experiment.sh molgan050 0
```

The reproduction workflow consists of:

1. model training;
2. generation of 10,000 peptide sequences and the main predictor-based evaluations;
3. novelty and diversity evaluation relative to the training dataset; and
4. PepSySco evaluation.

The main evaluation includes DeepImmuno-CNN, IEDB Class I immunogenicity, NetMHCpan 4.0, NetMHCpan 4.1, PEPMatch, and the similarity analysis. PepSySco and the detailed novelty analysis are subsequently performed using their corresponding evaluation scripts.

TCGA-BLCA exact-match evaluation can additionally be enabled by providing the prepared TCGA-BLCA peptide dataset through the `TCGA_CSV` environment variable.

### Reproducing the main multi-seed experiments

`reproduce_main_results.sh` reproduces the main experiments across random seeds 0--10:

```bash
./reproduce_main_results.sh
```

The script trains each unique model once for each seed. For the MolGAN variants, both `model_last.pth` and `model_best.pth` are evaluated from the same training run, without repeating model training.

The seed range can optionally be restricted. For example:

```bash
START_SEED=0 END_SEED=2 ./reproduce_main_results.sh
```

Because the complete multi-seed reproduction involves multiple GPU training runs and external predictor evaluations, the script should be executed in an appropriate compute environment rather than on a cluster login node.

### Reproducing the PepINVENT comparison

`reproduce_pepinvent.sh` reproduces the PepINVENT comparison experiment.

For the bladder-cancer experiment with random seed 0:

    ./reproduce_pepinvent.sh bladder 0

For the supplementary brain-cancer experiment:

    ./reproduce_pepinvent.sh brain 0

The PepINVENT/REINVENT environment should be active when running this script. Because DeepImmuno scoring uses the separate TensorFlow environment, `DEEPIMMUNO_PYTHON` must point to the Python executable from `environment_tf.yml`.

Additional details are provided in `code_R2_pepinvent/README.md`.

### Immunogenicity predictor checkpoint

The exact CNN immunogenicity-predictor checkpoint used in the experiments is provided in `weights/Immunogenicity_Predictor/`. The predictor uses the DeepImmuno-CNN architecture described by Li et al.

## Tables and plotting

The scripts and configuration files used to generate the manuscript tables are provided under `tables/`.

- `tables/Bladder/` contains the bladder-cancer table generators and configurations.
- `tables/Brain/` contains the supplementary brain-cancer table generators and configurations.

The final training-curve plotting code is provided under `plotting/`.

- `plotting/table1_curves_zoom.py` generates the training immunogenicity-score, unique-rate, zoomed unique-rate, and loss plots using `plotting/table1_config.txt`.
- `plotting/ablation/Plot_score_GD-WGAN-GP_effects.py` generates the GD-WGAN-GP ablation training curves.

## Third-party software

The Python package versions and source commits recorded from the development checkouts for REINVENT4 and PepFun2 are provided in `THIRD_PARTY_VERSIONS.md`. The PepINVENT prior used in the experiments is provided in `code_R2_pepinvent/priors/`.

