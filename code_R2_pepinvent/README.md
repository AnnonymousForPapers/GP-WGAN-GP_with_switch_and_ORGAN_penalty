# PepINVENT comparison

This directory contains the experiment-specific code used for the PepINVENT comparison in the manuscript.

## Software

The PepINVENT experiments use:

- `reinvent==4.8.24`
- `pepfun==2.0`

The corresponding source commits used during development are recorded in `../THIRD_PARTY_VERSIONS.md`.

The full REINVENT4 and PepFun2 source repositories are not bundled in this release. The required experiment-specific PepINVENT prior is provided at:

`priors/pepinvent.prior`

Its SHA-256 checksum is provided in:

`priors/SHA256SUMS.txt`

## Environments

The PepINVENT environment is specified by:

`../environment_pepinvent.yml`

DeepImmuno scoring uses the separate TensorFlow environment specified by:

`../environment_tf.yml`

Before running the PepINVENT experiment, set `DEEPIMMUNO_PYTHON` to the Python executable from the TensorFlow environment. For example:

    export DEEPIMMUNO_PYTHON=/path/to/tf/environment/bin/python

The PepINVENT/REINVENT environment should be active when running the experiment.

## Reproduction

From the repository root, run:

    ./reproduce_pepinvent.sh bladder 0

for the bladder-cancer experiment with random seed 0.

For the supplementary brain-cancer experiment:

    ./reproduce_pepinvent.sh brain 0

The launcher uses the released prior at `code_R2_pepinvent/priors/pepinvent.prior`.

The default experimental settings are 1,000 training epochs, checkpoint evaluation every 50 epochs, three masked residues per input peptide, and one masked input per source peptide. The random seed is supplied through the command line.

The final checkpoint is saved as `model_last.chkpt`. Checkpoints are also evaluated periodically, and the best checkpoint is saved as `model_best.chkpt` using the sum of the mean predicted immunogenicity score and the unique-peptide ratio.

## Runtime adaptation

The following files provide the experiment-specific runtime behavior required by the PepINVENT implementation:

- `pepinvent_runtime_env.py`
- `pepinvent_runtime_overrides.py`
- `sitecustomize.py`

These runtime adaptations avoid requiring modification of the installed REINVENT4 source tree.

## Main files

- `run_pepinvent_bladder3mask_epochs.py`
- `run_pepinvent_brain3mask_epochs.py`
- `run_pepinvent_inference_random3mask.py`
- `pepinvent_chuckles.py`
- `pepinvent_deepimmuno_direct_score.py`
- `pepinvent_deepimmuno_with_scorer_score.py`
- `deepimmuno_tf_score_sequences.py`
- `evaluate_pepinvent_checkpoint.py`
- `evaluate_pepinvent_checkpoint_tf.py`
- `pepinvent_runtime_env.py`
- `pepinvent_runtime_overrides.py`
- `sitecustomize.py`

The DeepImmuno predictor weights used by these experiments are provided at:

`../weights/Immunogenicity_Predictor/`
