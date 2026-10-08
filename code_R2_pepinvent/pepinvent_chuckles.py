#!/usr/bin/env python3
"""
Natural20 PepINVENT CHUCKLES utilities.

This revision does NOT require OpenBabel.

The canonical CHUCKLES strings for the 20 natural amino acids are stored
directly below.  These are the same internal residue forms observed in the
PepINVENT outputs/logs, with a terminal O appended for the full amino-acid
form used by the input-construction helper.

Exactly three residue positions can be masked with "?".
"""

from pathlib import Path

import numpy as np
import pandas as pd


NATURAL_AA = set("ARNDCQEGHILKMFPSTWYV")

# Canonical PepINVENT/CHUCKLES internal residue fragments.
#
# Internal form is what appears inside a linear peptide:
#     ...C(=O)
#
# Full form used by build_linear_masked_peptide() has one terminal O:
#     ...C(=O)O
#
# This fixed table removes the runtime dependency on Python OpenBabel.
AA_INTERNAL_CHUCKLES = {
    "A": "N[C@@H](C)C(=O)",
    "R": "N[C@@H](CCCNC(=N)N)C(=O)",
    "N": "N[C@@H](CC(=O)N)C(=O)",
    "D": "N[C@@H](CC(=O)O)C(=O)",
    "C": "N[C@@H](CS)C(=O)",
    "Q": "N[C@@H](CCC(=O)N)C(=O)",
    "E": "N[C@@H](CCC(=O)O)C(=O)",
    "G": "NCC(=O)",
    "H": "N[C@@H](Cc1c[nH]cn1)C(=O)",
    "I": "N[C@@H]([C@H](CC)C)C(=O)",
    "L": "N[C@@H](CC(C)C)C(=O)",
    "K": "N[C@@H](CCCCN)C(=O)",
    "M": "N[C@@H](CCSC)C(=O)",
    "F": "N[C@@H](Cc1ccccc1)C(=O)",
    "P": "N1[C@@H](CCC1)C(=O)",
    "S": "N[C@@H](CO)C(=O)",
    "T": "N[C@@H]([C@H](C)O)C(=O)",
    "W": "N[C@@H](Cc1c[nH]c2ccccc12)C(=O)",
    "Y": "N[C@@H](Cc1ccc(O)cc1)C(=O)",
    "V": "N[C@@H](C(C)C)C(=O)",
}

AA_CHUCKLES = {
    aa: fragment + "O"
    for aa, fragment in AA_INTERNAL_CHUCKLES.items()
}


def aa_chuckles_table():
    """Return a copy of the fixed Natural20 full CHUCKLES table."""
    if set(AA_CHUCKLES) != NATURAL_AA:
        raise RuntimeError(
            "Natural20 CHUCKLES table keys do not match the 20 natural amino acids."
        )
    if len(set(AA_CHUCKLES.values())) != 20:
        raise RuntimeError(
            "Natural20 CHUCKLES table does not contain 20 unique fragments."
        )
    return dict(AA_CHUCKLES)


def build_linear_masked_peptide(sequence: str, mask_positions):
    """
    sequence: natural 9/10-mer.
    mask_positions: zero-based positions to replace by "?".

    Returns:
        original_peptide_smiles, pepinvent_input

    The formatting follows the same PepINVENT linear-peptide convention:
      - non-final residues use the internal fragment (...C(=O))
      - the final residue retains one terminal O (...C(=O)O)
      - residues are separated with "|"
    """
    sequence = sequence.strip().upper()

    if len(sequence) not in (9, 10):
        raise ValueError(
            f"Expected 9/10-mer, got length={len(sequence)}: {sequence}"
        )

    if not set(sequence).issubset(NATURAL_AA):
        raise ValueError(f"Non-natural character in peptide: {sequence}")

    mask_positions = sorted(set(int(x) for x in mask_positions))

    if any(x < 0 or x >= len(sequence) for x in mask_positions):
        raise ValueError(
            f"Mask position outside sequence: {mask_positions}"
        )

    # Start from the full residue form (...C(=O)O).
    fragments = [AA_CHUCKLES[aa] for aa in sequence]

    # Match the prior implementation's linear-terminal convention:
    # append an extra O to the final full residue, then remove the final
    # character from each unmasked fragment during merge.
    fragments[-1] = fragments[-1] + "O"

    masked = [
        "?" if i in mask_positions else frag
        for i, frag in enumerate(fragments)
    ]

    pepinvent_input = "|".join(
        frag if frag == "?" else frag[:-1]
        for frag in masked
    )

    original_smiles = "".join(
        frag[:-1] for frag in fragments
    )

    return original_smiles, pepinvent_input



def decode_natural20_linear_smiles(smiles: str):
    """
    Deterministically decode a linear Natural20 peptide SMILES/CHUCKLES string
    produced from AA_INTERNAL_CHUCKLES.

    This avoids relying on PepFun's generic SMILES->peptide recognition.

    Accepted terminal conventions:
      1) final residue ends in ...C(=O)O
      2) final residue ends in ...C(=O)

    Returns a 9/10-mer sequence or None.
    """
    if smiles is None:
        return None

    s = str(smiles).strip()
    if not s:
        return None

    # Generated PepINVENT linear peptides are concatenations of the same
    # canonical internal residue fragments used by constrained decoding.
    # The last residue may carry one extra terminal O.
    candidates = [s]
    if s.endswith("O"):
        candidates.append(s[:-1])

    items = list(AA_INTERNAL_CHUCKLES.items())
    solutions = []

    for candidate in candidates:
        memo = {}

        def rec(pos, count):
            key = (pos, count)
            if key in memo:
                return memo[key]

            if pos == len(candidate):
                return [""]

            if count >= 10:
                return []

            out = []
            for aa, fragment in items:
                if candidate.startswith(fragment, pos):
                    tails = rec(
                        pos + len(fragment),
                        count + 1,
                    )
                    for tail in tails:
                        out.append(aa + tail)

            memo[key] = out
            return out

        for seq in rec(0, 0):
            if len(seq) in (9, 10):
                solutions.append(seq)

    # De-duplicate while preserving order.
    solutions = list(dict.fromkeys(solutions))

    if len(solutions) == 1:
        return solutions[0]

    # Zero matches -> not representable by this exact Natural20 grammar.
    # Multiple matches -> ambiguous; do not guess.
    return None


def validate_natural20_sequence(seq):
    if seq is None:
        return None
    seq = str(seq).strip().upper()
    if len(seq) not in (9, 10):
        return None
    if not set(seq).issubset(NATURAL_AA):
        return None
    return seq


def detect_peptide_column(df: pd.DataFrame):
    candidates = [
        "peptide", "Peptide", "PEPTIDE",
        "sequence", "Sequence", "SEQUENCE",
        "epitope", "Epitope",
        "mut_peptide", "Mut_peptide", "MT_pep", "mutant_peptide",
    ]
    for c in candidates:
        if c in df.columns:
            return c

    # Fallback: choose the column containing the largest number of valid
    # natural 9/10-mers.
    best_col = None
    best_count = 0
    for c in df.columns:
        vals = df[c].dropna().astype(str).str.strip().str.upper()
        count = sum(
            len(x) in (9, 10) and set(x).issubset(NATURAL_AA)
            for x in vals
        )
        if count > best_count:
            best_col = c
            best_count = count

    if best_col is None or best_count == 0:
        raise RuntimeError(
            "Could not identify a column containing natural 9/10-mer peptides. "
            f"Columns: {list(df.columns)}"
        )
    return best_col


def load_peptides(csv_path):
    df = pd.read_csv(csv_path)
    peptide_col = detect_peptide_column(df)

    peptides = []
    for x in df[peptide_col].dropna().astype(str):
        seq = x.strip().upper()
        if len(seq) in (9, 10) and set(seq).issubset(NATURAL_AA):
            peptides.append(seq)

    # Preserve first occurrence while removing duplicates.
    peptides = list(dict.fromkeys(peptides))
    if not peptides:
        raise RuntimeError(f"No valid natural 9/10-mers found in {csv_path}")

    return peptides, peptide_col


def prepare_all_training_masks(
    csv_path,
    output_smi,
    output_map_csv,
    seed,
    mask_count=3,
    masks_per_peptide=1,
):
    peptides, peptide_col = load_peptides(csv_path)
    rng = np.random.default_rng(seed)

    rows = []
    smi_rows = []

    for source_idx, peptide in enumerate(peptides):
        seen_masks = set()
        target = int(masks_per_peptide)
        attempts = 0

        while len(seen_masks) < target:
            attempts += 1
            if attempts > 1000:
                raise RuntimeError(
                    f"Could not generate {target} unique masks for {peptide}"
                )

            positions = tuple(sorted(
                rng.choice(len(peptide), size=mask_count, replace=False).tolist()
            ))
            if positions in seen_masks:
                continue
            seen_masks.add(positions)

            original_smiles, masked_input = build_linear_masked_peptide(
                peptide, positions
            )
            smi_rows.append(masked_input)
            rows.append({
                "source_index": source_idx,
                "source_peptide": peptide,
                "peptide_length": len(peptide),
                "masked_positions_0based": ",".join(map(str, positions)),
                "masked_positions_1based": ",".join(str(x + 1) for x in positions),
                "masked_input": masked_input,
                "original_peptide_smiles": original_smiles,
            })

    Path(output_smi).write_text("\n".join(smi_rows) + "\n", encoding="utf-8")
    pd.DataFrame(rows).to_csv(output_map_csv, index=False)

    return {
        "peptide_column": peptide_col,
        "num_unique_source_peptides": len(peptides),
        "num_masked_inputs": len(rows),
    }

def load_bladder_peptides(csv_path):
    df = pd.read_csv(csv_path)
    peptide_col = detect_peptide_column(df)

    peptides = []
    for x in df[peptide_col].dropna().astype(str):
        seq = x.strip().upper()
        if len(seq) in (9, 10) and set(seq).issubset(NATURAL_AA):
            peptides.append(seq)

    # Preserve first occurrence while removing duplicates.
    peptides = list(dict.fromkeys(peptides))
    if not peptides:
        raise RuntimeError(f"No valid natural 9/10-mers found in {csv_path}")

    return peptides, peptide_col


def prepare_random_masks(
    peptides,
    n,
    output_smi,
    output_map_csv,
    seed,
    mask_count=3,
):
    rng = np.random.default_rng(seed)
    rows = []
    smi_rows = []

    for sample_idx in range(int(n)):
        source_idx = int(rng.integers(0, len(peptides)))
        peptide = peptides[source_idx]
        positions = tuple(sorted(
            rng.choice(len(peptide), size=mask_count, replace=False).tolist()
        ))

        original_smiles, masked_input = build_linear_masked_peptide(
            peptide, positions
        )

        smi_rows.append(masked_input)
        rows.append({
            "sample_index": sample_idx,
            "source_index": source_idx,
            "source_peptide": peptide,
            "peptide_length": len(peptide),
            "masked_positions_0based": ",".join(map(str, positions)),
            "masked_positions_1based": ",".join(str(x + 1) for x in positions),
            "masked_input": masked_input,
            "original_peptide_smiles": original_smiles,
        })

    Path(output_smi).write_text("\n".join(smi_rows) + "\n", encoding="utf-8")
    pd.DataFrame(rows).to_csv(output_map_csv, index=False)
