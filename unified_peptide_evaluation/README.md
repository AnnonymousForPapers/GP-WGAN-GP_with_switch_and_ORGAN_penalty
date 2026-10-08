# Unified peptide evaluation pipeline

This package consolidates the uploaded generation/evaluation scripts into one seed-0 pipeline for:

- residual CNN GAN / WGAN generator checkpoints (`.pth`)
- LSTM generator checkpoints (`.pth`)
- unmasked RoPE/RMSNorm Transformer generator checkpoints (`.pth`)
- D3PM + attached Transformer checkpoints (`.pth`)
- PepINVENT / REINVENT checkpoints (`.chkpt`)

The architecture-specific code ends after peptide generation. Every model is then normalized to the same natural 9/10-mer table, so all predictors use the same downstream evaluation code.

## Main command

```bash
python run_all_evaluations.py \
  --model-dir result/Goal-directed_WGAN-GP_ORGAN_FixPad_seed0/epoch1000 \
  --checkpoint best \
  --seed 0 \
  --num-samples 10000
```

`--checkpoint auto` searches in this order:

1. `model_best.pth` / `model_best.chkpt`
2. `model_last.pth` / `model_last.chkpt`
3. highest numbered `model_epoch_*.pth` / `.chkpt`

For PyTorch checkpoints, `--architecture auto` detects GAN, LSTM, Transformer, or D3PM from state-dict keys. PepINVENT is recognized from `.chkpt` unless `--architecture` is set explicitly.

## Stages

The runner prints and logs completion of each stage:

1. Generate/load peptides using fixed seed 0.
2. Preserve peptides with fewer than two `-` placeholders, remove the remaining placeholder, validate natural 9/10-mers, and deduplicate.
3. DeepImmuno.
4. IEDB Class-I pMHC immunogenicity through the IEDB Next-Generation API.
5. NetMHCpan 4.0 BA.
6. NetMHCpan 4.1 BA.
7. IEDB PEPMatch against the built-in Human proteome, using `mismatch: 3` and best-match-per-peptide.
8. IEDB PepSySco synthesis-success score.
9. Exact-match comparison against TCGA-BLCA mutant peptides.
10. Similarity to `Bladder.4.0_test_mut.csv` using the same `difflib.SequenceMatcher` measure as the uploaded script.
11. Merge outputs, make IC50 summaries/plots, and write the final summary.

Predictor failures are isolated by default: the pipeline logs the failure and continues. Add `--strict` to terminate on a predictor failure.

## Output

By default results are placed inside the model directory as `evaluation_<checkpoint>_seed0/` and include:

```text
generated_raw.csv
generated_raw_seed0_batch10000.txt
generated_filtered_all.csv
generated_filtered_unique.csv
generated_filtered_unique_seed0_batch10000.txt
all_predictions.csv
summary.json
summary.csv
predictor_manifest.json
evaluation.log

deepimmuno/
  deepimmuno.csv
  deepimmuno_scored.txt

iedb_immunogenicity/
  IEDB_immunogenicity_prediction.csv
  IEDB_immunogenicity_raw_combined.csv
  raw_api/*.json

netmhcpan_4_0/
  NetMHCpan_4.0_prediction.csv
  raw_api/*.tsv

netmhcpan_4_1/
  NetMHCpan_4.1_prediction.csv
  raw_api/*.tsv

pepmatch/
  PEPMatch_prediction.csv
  PEPMatch_raw_combined.csv
  raw_api/*.json

pepsysco/
  PepSySco_prediction.csv
  PepSySco_raw_combined.csv
  raw_api/*.json

tcga_blca/
  TCGA_BLCA_prediction.csv
  generated_vs_TCGA_BLCA_exact_matches.csv
  generated_vs_TCGA_BLCA_all.csv
  generated_vs_TCGA_BLCA_unmatched.csv
  generated_vs_TCGA_BLCA_summary.txt

bladder_similarity/
  similarity.csv
  MaxSimilarity.npy
  similarity_top5.txt

IC50_compare_NetMHCpan_4.0.png
IC50_compare_NetMHCpan_4.1.png
```

The saved NetMHCpan prediction files include the compatibility column `Aff(nM)` as well as version-specific columns.

## DeepImmuno

The worker uses the uploaded DeepImmuno architecture and **FixPad position 5** for 9-mers:

```python
peptide = peptide[:5] + "-" + peptide[5:]
```

It auto-searches for a root containing:

```text
data/DeepImmuno/after_pca.txt
data/DeepImmuno/hla2paratopeTable_aligned.txt
weights/Immunogenicity_Predictor/
```

Override with:

```bash
--data-root .
```

The DeepImmuno weight loader now accepts either a weight/checkpoint file or a directory.
For the original DeepImmuno release layout (`.index` + `.data-*` inside a directory),
it automatically creates temporary non-dot checkpoint names before calling Keras,
which avoids `IsADirectoryError` on newer Keras versions.

If your weights live outside `--data-root`, point to them explicitly:

```bash
--deepimmuno-weights weights/Immunogenicity_Predictor
```

You may also pass a specific `.h5`, `.hdf5`, `.keras`, TensorFlow checkpoint prefix,
or `.index` file.

The default TensorFlow Python is:

```text
/path/to/tf/environment/bin/python
```

Override with `--tf-python`.

## NetMHCpan 4.0 and 4.1

The default backend is the official IEDB MHC-I REST service and uses NetMHCpan binding-affinity mode. No local NetMHCpan installation is required if the compute node has outbound HTTPS access.

```bash
--netmhcpan-backend api
```

For local installations:

```bash
--netmhcpan-backend local \
--netmhcpan40-exe /path/to/netMHCpan-4.0/netMHCpan \
--netmhcpan41-exe /path/to/netMHCpan-4.1/netMHCpan
```

The final summary reports the same affinity bins used by the old evaluation:

- `IC50 < 150 nM`
- `150 <= IC50 < 500 nM`
- `IC50 >= 500 nM`
- total `IC50 < 500 nM`


## PEPMatch (Human proteome, up to 3 substitutions)

PEPMatch is run through the IEDB Next-Generation API with direct peptide-sequence input. The default request is:

```json
{
  "mismatch": 3,
  "proteome": "Human",
  "best_match": true
}
```

No local FASTA file is required. The primary merged result keeps one best human-proteome match per generated peptide and adds:

```text
pepmatch_matched_sequence
pepmatch_protein_id
pepmatch_protein_name
pepmatch_gene
pepmatch_mismatches
pepmatch_mutated_positions
pepmatch_exact_match
pepmatch_within_1_mismatch
pepmatch_within_2_mismatches
pepmatch_within_3_mismatches
pepmatch_no_match_within_3
```

The mismatch threshold can still be overridden with `--pepmatch-mismatch 0..5`, but the package default is now **3**.

## PepSySco

PepSySco is run through the IEDB Next-Generation API and receives the same filtered natural peptide sequences directly. It adds:

```text
pepsysco_score
```

The score is retained as a continuous 0-1 value; the pipeline does not impose a pass/fail cutoff. Raw IEDB request/submission/result JSON files are retained for reproducibility.

IEDB Next-Generation API options shared by IEDB immunogenicity, PEPMatch, and PepSySco:

```bash
--iedb-nextgen-api-base https://api-nextgen-tools.iedb.org/api/v1 \
--iedb-nextgen-timeout 900 \
--iedb-nextgen-poll-seconds 2
```

## IEDB Class-I immunogenicity

The default backend is now the **IEDB Next-Generation T-cell Class I API**. No standalone `predict_immunogenicity.py` installation is required when the compute node has outbound HTTPS access.

The evaluator submits the already-filtered peptide sequences directly, with:

```json
{
  "alleles": "HLA-A*02:01",
  "peptide_length_range": null,
  "predictors": [
    {
      "type": "immunogenicity",
      "mask_choice": "default"
    }
  ]
}
```

`peptide_length_range: null` is deliberate: the input rows are already 9/10-mer peptides, so the API evaluates them as-is rather than tiling them into new subsequences.

Default masking uses IEDB's default Class-I anchor masking. The backend and masking can be changed with:

```bash
--iedb-immunogenicity-backend api \
--iedb-immunogenicity-mask-choice default \
--iedb-immunogenicity-chunk-size 500
```

For a custom mask:

```bash
--iedb-immunogenicity-mask-choice custom \
--iedb-immunogenicity-position-to-mask 2,5,9
```

The output adds:

```text
iedb_immunogenicity_score
iedb_immunogenicity_allele
```

and retains the raw IEDB request/submission/result JSON files under `iedb_immunogenicity/raw_api/`.

The legacy standalone tool is retained only as an optional offline fallback:

```bash
--iedb-immunogenicity-backend local \
--iedb-immunogenicity-script /path/to/predict_immunogenicity.py \
--iedb-immunogenicity-python /path/to/python2
```

`--iedb-immunogenicity-backend auto` tries the API first and uses the local fallback only if it has been configured.


## TCGA-BLCA exact mutant-peptide comparison

The TCGA stage uses the **same unique, valid post-filter 9/10-mer peptide list** sent to the other predictors. It performs an exact sequence merge against the `mutant_peptide` column of the TCGA-BLCA reference, matching the uploaded standalone comparison script.

The reference CSV must contain at least:

```text
wt_peptide
mutant_peptide
peptide_length
```

When present, the following annotations are preserved in the detailed match table and merged into compact per-peptide columns:

```text
mutation_position_in_peptide
genes
mutations
n_patients
n_samples
patient_ids
sample_ids
protein_accessions
```

The easiest location is:

```text
<data-root>/data/TCGA/TCGA_BLCA_WT_mutant_peptides_unique.csv
```

The runner also searches a few common project locations automatically. You can always specify the file explicitly:

```bash
--tcga-csv /path/to/TCGA_BLCA_WT_mutant_peptides_unique.csv
```

The compact columns added to `all_predictions.csv` include:

```text
tcga_blca_exact_match
tcga_blca_match_rows
tcga_blca_wt_peptides
tcga_blca_mutant_peptides
tcga_blca_genes
tcga_blca_mutations
tcga_blca_mutation_positions
tcga_blca_patient_ids
tcga_blca_sample_ids
tcga_blca_protein_accessions
tcga_blca_unique_patients
tcga_blca_unique_samples
```

The detailed exact-match CSV keeps every TCGA reference row, so one generated peptide can map to multiple mutation/patient annotations without losing information. The summary reports the number and percentage of unique generated peptides with at least one exact TCGA-BLCA mutant-peptide match, separately including matched 9-mers and 10-mers.

## PepINVENT

For a PepINVENT checkpoint:

```bash
python run_all_evaluations.py \
  --model-dir /path/to/PepINVENT_DeepImmuno_direct_seed0/epoch1000 \
  --checkpoint best \
  --architecture pepinvent \
  --reinvent reinvent \
  --seed 0 \
  --num-samples 10000
```

The package contains the uploaded deterministic Natural20 CHUCKLES encoder/decoder. Your training workflow additionally used runtime-only Natural20 sampling hooks (`sitecustomize.py` + `pepinvent_runtime_overrides.py`). For exact checkpoint-comparable sampling, point the evaluator at the directory containing those files:

```bash
--pepinvent-runtime-dir /path/to/code_R2_pepinvent
```

The runner tries to discover that directory automatically below the data/model roots.

## D3PM

The number of diffusion steps is inferred from `time_embedding.weight`, so checkpoints from T=125, 250, 500, or 1000 do not require separate flags. D3PM sampling defaults to batches of 64 because every sample traverses all reverse steps:

```bash
--d3pm-batch-size 64
```

## Evaluate an already-generated peptide file

Generation can be skipped completely:

```bash
python run_all_evaluations.py \
  --input-peptides my_generated_peptides.csv \
  --seed 0
```

## Select predictors

All:

```bash
--predictors all
```

No predictors (generation/filtering only):

```bash
--predictors none
```

Subset:

```bash
--predictors deepimmuno iedb_immunogenicity netmhcpan40 netmhcpan41 pepmatch pepsysco tcga_blca similarity
```

Available built-ins are:

```text
deepimmuno
iedb_immunogenicity
netmhcpan40
netmhcpan41
pepmatch
pepsysco
tcga_blca
similarity
```

## Adding another predictor

Generation code does not need to change. Copy `unified_peptide_eval/predictors/custom_example.py`, subclass `BasePredictor`, and implement:

```python
def predict(self, peptides, context):
    return pandas.DataFrame({
        "peptide": peptides,
        "my_new_score": scores,
    })
```

Then expose it in `predictors/__init__.py` and add one factory branch in `make_predictor()` in `run_all_evaluations.py`. Local Python models, command-line programs, and HTTP APIs can all use this interface.

## Dependencies

Core:

```bash
pip install numpy pandas torch matplotlib
```

DeepImmuno runs in the TensorFlow environment rather than forcing TensorFlow into the main process. PepINVENT requires your existing REINVENT4 environment. IEDB immunogenicity, PEPMatch, and PepSySco use the IEDB Next-Generation HTTPS API by default, so they do not require separate local predictor installations.

## Reproducibility

The evaluation seed defaults to `0` and seeds Python, NumPy, PyTorch CPU/CUDA. `predictor_manifest.json` records predictor versions/backends and `evaluation.log` records completed/failed stages.


## PepSySco API limitation (Sep 2026)

The IEDB website provides PepSySco, and the NG API exposes `GET /api/v1/pepsysco` parameter metadata. However, the public `POST /api/v1/pipeline` endpoint currently rejects `tool_group="pepsysco"`. Therefore this package no longer attempts an unsupported pipeline submission.

With `--pepsysco-backend auto` (default), PepSySco is **skipped** unless one of these is configured:

```bash
--pepsysco-results-csv /path/to/Pepsysco_download.csv
```

Use this after running the same filtered peptide list through the IEDB PepSySco web tool and downloading its CSV. The pipeline merges the `peptide` and PepSySco score columns into `all_predictions.csv`.

Or configure a local standalone implementation:

```bash
--pepsysco-backend command \
--pepsysco-command 'python /path/to/local_pepsysco.py --input {input} --output {output}'
```

The command must write a CSV containing a peptide/sequence column and a PepSySco/score column.
