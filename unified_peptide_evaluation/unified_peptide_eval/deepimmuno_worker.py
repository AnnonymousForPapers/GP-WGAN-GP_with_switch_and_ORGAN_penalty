#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shutil
import tempfile
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd
from tensorflow import keras, train as tf_train
from tensorflow.keras import layers

AA_ORDER = "ARNDCQEGHILKMFPSTWYV-"
HLA_NAME = "HLA-A*0201"


def build_model():
    input1 = keras.Input(shape=(10, 12, 1))
    input2 = keras.Input(shape=(46, 12, 1))
    x = layers.Conv2D(filters=16, kernel_size=(2, 12))(input1)
    x = layers.BatchNormalization()(x); x = keras.activations.relu(x)
    x = layers.Conv2D(filters=32, kernel_size=(2, 1))(x)
    x = layers.BatchNormalization()(x); x = keras.activations.relu(x)
    x = layers.MaxPool2D(pool_size=(2, 1), strides=(2, 1))(x)
    x = layers.Flatten()(x); x = keras.Model(inputs=input1, outputs=x)

    y = layers.Conv2D(filters=16, kernel_size=(15, 12))(input2)
    y = layers.BatchNormalization()(y); y = keras.activations.relu(y)
    y = layers.MaxPool2D(pool_size=(2, 1), strides=(2, 1))(y)
    y = layers.Conv2D(filters=32, kernel_size=(9, 1))(y)
    y = layers.BatchNormalization()(y); y = keras.activations.relu(y)
    y = layers.MaxPool2D(pool_size=(2, 1), strides=(2, 1))(y)
    y = layers.Flatten()(y); y = keras.Model(inputs=input2, outputs=y)

    z = layers.concatenate([x.output, y.output])
    z = layers.Dense(128, activation="relu")(z)
    z = layers.Dropout(0.2)(z)
    z = layers.Dense(1, activation="sigmoid")(z)
    return keras.Model(inputs=[input1, input2], outputs=z)


def _link_or_copy(src: Path, dst: Path):
    try:
        os.symlink(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def resolve_weights_path(data_root, explicit_weights=None):
    """Return (load_weights_path, temporary_directory_or_None).

    DeepImmuno's original released checkpoint has an unusual TensorFlow
    checkpoint prefix: the files inside the directory are named `.index` and
    `.data-00000-of-00001`.  TensorFlow 2.3 accepted the directory path used by
    the original DeepImmuno code, but newer Keras may send that directory to
    h5py and raise IsADirectoryError.  For that legacy layout we create
    temporary non-dot checkpoint names and load the equivalent checkpoint
    prefix without modifying the original files.
    """
    root = Path(data_root).expanduser().resolve()
    target = (Path(explicit_weights).expanduser().resolve()
              if explicit_weights else root / "weights/Immunogenicity_Predictor")

    if target.is_file():
        if target.suffix == ".index":
            return str(target)[:-len(".index")], None
        return str(target), None

    if not target.exists():
        raise FileNotFoundError(f"DeepImmuno weights path does not exist: {target}")
    if not target.is_dir():
        raise RuntimeError(f"DeepImmuno weights path is neither file nor directory: {target}")

    # Original DeepImmuno release layout: `.index` + `.data-*`.
    dot_index = target / ".index"
    dot_data = sorted(target.glob(".data-*"))
    if dot_index.exists() and dot_data:
        holder = tempfile.TemporaryDirectory(prefix="deepimmuno_weights_")
        prefix = Path(holder.name) / "deepimmuno_ckpt"
        _link_or_copy(dot_index, Path(str(prefix) + ".index"))
        for shard in dot_data:
            # shard.name is e.g. '.data-00000-of-00001'
            _link_or_copy(shard, Path(str(prefix) + shard.name))
        return str(prefix), holder

    # Standard TensorFlow checkpoint directory.
    latest = tf_train.latest_checkpoint(str(target))
    if latest and not Path(latest).is_dir():
        return latest, None

    # Checkpoint shards may be present without a TensorFlow 'checkpoint' state file.
    index_files = sorted(target.rglob("*.index"), key=lambda x: x.stat().st_mtime, reverse=True)
    if index_files:
        index_file = index_files[0]
        if index_file.name == ".index":
            # Same legacy layout, possibly in a nested directory.
            legacy_dir = index_file.parent
            shards = sorted(legacy_dir.glob(".data-*"))
            if shards:
                holder = tempfile.TemporaryDirectory(prefix="deepimmuno_weights_")
                prefix = Path(holder.name) / "deepimmuno_ckpt"
                _link_or_copy(index_file, Path(str(prefix) + ".index"))
                for shard in shards:
                    _link_or_copy(shard, Path(str(prefix) + shard.name))
                return str(prefix), holder
        return str(index_file)[:-len(".index")], None

    # Keras/HDF5 formats. Prefer the newest candidate if several are present.
    file_candidates = []
    for pattern in ("*.weights.h5", "*.h5", "*.hdf5", "*.keras"):
        file_candidates.extend(target.rglob(pattern))
    file_candidates = list({x.resolve(): x for x in file_candidates}.values())
    if file_candidates:
        file_candidates.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        return str(file_candidates[0]), None

    entries = sorted(str(x.relative_to(target)) for x in target.rglob("*") if x.is_file())[:30]
    shown = "\n  ".join(entries) if entries else "(directory is empty)"
    raise FileNotFoundError(
        "Could not find DeepImmuno model weights inside " + str(target) +
        ". Expected the original .index/.data-* checkpoint, a standard TensorFlow checkpoint, "
        "or a .h5/.hdf5/.keras weight file. Files found:\n  " + shown
    )


class DeepImmuno:
    def __init__(self, data_root, weights_path=None):
        root = Path(data_root).expanduser().resolve()
        self.after_pca = np.loadtxt(root / "data/DeepImmuno/after_pca.txt")
        hla = pd.read_csv(root / "data/DeepImmuno/hla2paratopeTable_aligned.txt", sep="\t")
        self.hla_dic = dict(zip(hla["HLA"], hla["pseudo"]))
        if HLA_NAME not in self.hla_dic:
            raise KeyError(f"{HLA_NAME} not present in HLA table")
        self.model = build_model()
        resolved_weights, self._weights_tmpdir = resolve_weights_path(root, weights_path)
        print(f"Resolved DeepImmuno weights: {resolved_weights}", flush=True)
        self.model.load_weights(resolved_weights)
        self.hla_encoded = self._encode_sequence(self.hla_dic[HLA_NAME])

    def _encode_sequence(self, sequence):
        matrix = np.transpose(self.after_pca)
        out = np.empty((len(sequence), 12), dtype=np.float32)
        for i, aa in enumerate(sequence.upper()):
            if aa == "X": aa = "-"
            out[i] = matrix[:, AA_ORDER.index(aa)]
        return out.reshape(len(sequence), 12, 1)

    def _encode_peptide(self, peptide):
        p = peptide.upper()
        if len(p) == 9:
            # Corrected FixPad convention used in the user's newer code.
            p = p[:5] + "-" + p[5:]
        elif len(p) != 10:
            raise ValueError(f"DeepImmuno requires 9/10-mer, got {peptide}")
        return self._encode_sequence(p)

    def score(self, peptides, batch_size=1024):
        if not peptides:
            return np.asarray([], dtype=np.float32)
        pep_x = np.stack([self._encode_peptide(p) for p in peptides], axis=0)
        hla_x = np.repeat(self.hla_encoded[None, ...], len(peptides), axis=0)
        return self.model.predict([pep_x, hla_x], batch_size=batch_size, verbose=0).reshape(-1).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--weights", default=None, help="Optional DeepImmuno weight file, checkpoint prefix, .index file, or directory.")
    args = ap.parse_args()
    df = pd.read_csv(args.input)
    peptides = df["peptide"].astype(str).tolist()
    scores = DeepImmuno(args.data_root, args.weights).score(peptides, args.batch_size)
    pd.DataFrame({
        "peptide": peptides,
        "HLA": [HLA_NAME] * len(peptides),
        "deepimmuno_score": scores,
    }).to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
