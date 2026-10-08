#!/usr/bin/env python3
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow.keras as keras
from tensorflow.keras import layers

AA_ORDER = "ARNDCQEGHILKMFPSTWYV-"
HLA_NAME = "HLA-A*0201"


def log(message):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"{ts} [DeepImmuno-TF] {message}", file=sys.stderr, flush=True)


def build_deepimmuno():
    input1 = keras.Input(shape=(10, 12, 1))
    input2 = keras.Input(shape=(46, 12, 1))

    x = layers.Conv2D(filters=16, kernel_size=(2, 12))(input1)
    x = layers.BatchNormalization()(x)
    x = keras.activations.relu(x)
    x = layers.Conv2D(filters=32, kernel_size=(2, 1))(x)
    x = layers.BatchNormalization()(x)
    x = keras.activations.relu(x)
    x = layers.MaxPool2D(pool_size=(2, 1), strides=(2, 1))(x)
    x = layers.Flatten()(x)
    x = keras.Model(inputs=input1, outputs=x)

    y = layers.Conv2D(filters=16, kernel_size=(15, 12))(input2)
    y = layers.BatchNormalization()(y)
    y = keras.activations.relu(y)
    y = layers.MaxPool2D(pool_size=(2, 1), strides=(2, 1))(y)
    y = layers.Conv2D(filters=32, kernel_size=(9, 1))(y)
    y = layers.BatchNormalization()(y)
    y = keras.activations.relu(y)
    y = layers.MaxPool2D(pool_size=(2, 1), strides=(2, 1))(y)
    y = layers.Flatten()(y)
    y = keras.Model(inputs=input2, outputs=y)

    combined = layers.concatenate([x.output, y.output])
    z = layers.Dense(128, activation="relu")(combined)
    z = layers.Dropout(0.2)(z)
    z = layers.Dense(1, activation="sigmoid")(z)

    return keras.Model(inputs=[input1, input2], outputs=z)



def resolve_deepimmuno_weights(weights_location):
    """
    Resolve the DeepImmuno Keras/TensorFlow weight location while preserving
    compatibility with the original GAN code, which calls:

        model.load_weights("../weights/Immunogenicity_Predictor/")

    Supported layouts:
      1) Directory used as a TensorFlow checkpoint prefix, with hidden
         .index / .data-* files.  In this case return the directory WITH
         a trailing slash, exactly like the GAN code.
      2) Directory containing a TensorFlow 'checkpoint' state file.
      3) Directory containing one or more *.index checkpoint prefixes.
      4) Directory containing a single *.h5 / *.hdf5 / *.keras weight file.
      5) Direct file/prefix path.
    """
    p = Path(weights_location)

    log(f"Resolving DeepImmuno weights from {p.resolve()}")

    if p.is_file():
        log(f"Resolved DeepImmuno weights as direct file: {p}")
        return str(p)

    # TensorFlow checkpoint prefix can itself be represented by sibling
    # <prefix>.index and <prefix>.data-* files.
    if Path(str(p) + ".index").exists():
        log(f"Resolved DeepImmuno TensorFlow checkpoint prefix: {p}")
        return str(p)

    if not p.is_dir():
        raise FileNotFoundError(f"DeepImmuno weight location does not exist: {p}")

    # IMPORTANT: this is the layout matching the original GAN expression
    # model.load_weights(... + 'weights/Immunogenicity_Predictor/').
    hidden_index = p / ".index"
    hidden_data = sorted(p.glob(".data-*"))
    if hidden_index.exists() and hidden_data:
        resolved = str(p) + os.sep
        log(
            "Resolved GAN-style TensorFlow checkpoint directory prefix: "
            f"{resolved}"
        )
        return resolved

    # Standard TensorFlow checkpoint directory with checkpoint metadata.
    try:
        latest = tf.train.latest_checkpoint(str(p))
    except Exception:
        latest = None
    if latest:
        log(f"Resolved latest TensorFlow checkpoint: {latest}")
        return latest

    # Explicit *.index checkpoint prefixes.
    index_files = sorted(p.glob("*.index"))
    if index_files:
        # Prefer names containing Immunogenicity_Predictor, otherwise newest.
        preferred = [
            x for x in index_files
            if "Immunogenicity_Predictor" in x.name
        ]
        chosen = preferred[-1] if preferred else index_files[-1]
        resolved = str(chosen)[:-len(".index")]
        log(f"Resolved TensorFlow checkpoint prefix from .index: {resolved}")
        return resolved

    # HDF5/Keras formats.
    keras_files = []
    for pattern in ("*.weights.h5", "*.h5", "*.hdf5", "*.keras"):
        keras_files.extend(sorted(p.glob(pattern)))
    # Deduplicate while preserving order.
    seen = set()
    keras_files = [x for x in keras_files if not (str(x) in seen or seen.add(str(x)))]
    if keras_files:
        chosen = keras_files[-1]
        log(f"Resolved Keras/HDF5 weight file: {chosen}")
        return str(chosen)

    # Helpful failure diagnostics.
    entries = sorted(x.name for x in p.iterdir())
    preview = entries[:50]
    raise FileNotFoundError(
        "Could not resolve DeepImmuno weights inside directory "
        f"{p}. First entries: {preview}"
    )


class DeepImmuno:
    def __init__(self, data_root):
        t0 = time.perf_counter()
        self.data_root = Path(data_root)
        log(f"Initializing DeepImmuno from data_root={self.data_root}")
        self.after_pca = np.loadtxt(
            self.data_root / "data/DeepImmuno/after_pca.txt"
        )
        hla = pd.read_csv(
            self.data_root / "data/DeepImmuno/hla2paratopeTable_aligned.txt",
            sep="\t",
        )
        self.hla_dic = dict(zip(hla["HLA"], hla["pseudo"]))

        log("Building CNN architecture")
        self.model = build_deepimmuno()

        weights_location = self.data_root / "weights/Immunogenicity_Predictor"
        resolved_weights = resolve_deepimmuno_weights(weights_location)
        log(f"Loading CNN weights from resolved path: {resolved_weights}")
        self.model.load_weights(resolved_weights)
        log("CNN weights loaded successfully")

        self.hla_encoded = self._encode_sequence(self.hla_dic[HLA_NAME])
        log(f"DeepImmuno initialization complete in {time.perf_counter()-t0:.3f}s")

    def _encode_sequence(self, sequence):
        matrix = np.transpose(self.after_pca)
        encoded = np.empty((len(sequence), 12), dtype=np.float32)
        for i, residue in enumerate(sequence.upper()):
            if residue == "X":
                residue = "-"
            encoded[i] = matrix[:, AA_ORDER.index(residue)]
        return encoded.reshape(len(sequence), 12, 1)

    def _encode_peptide(self, peptide):
        peptide = peptide.upper()
        if len(peptide) == 9:
            peptide = peptide[:5] + "-" + peptide[5:]
        elif len(peptide) != 10:
            raise ValueError(f"Expected 9/10-mer, got {len(peptide)}")
        return self._encode_sequence(peptide)

    def score(self, peptides):
        if not peptides:
            return []

        log(f"Encoding {len(peptides)} peptide(s)")
        t0 = time.perf_counter()
        pep_x = np.stack([self._encode_peptide(p) for p in peptides], axis=0)
        hla_x = np.repeat(self.hla_encoded[None, ...], len(peptides), axis=0)
        log(f"Encoding complete in {time.perf_counter()-t0:.3f}s")

        log("Starting CNN prediction")
        t1 = time.perf_counter()
        pred = self.model.predict([pep_x, hla_x], verbose=0)
        log(f"CNN prediction complete in {time.perf_counter()-t1:.3f}s")

        return [float(x) for x in pred.reshape(-1)]


def main():
    if len(sys.argv) != 2:
        raise SystemExit("Usage: deepimmuno_tf_score_sequences.py DATA_ROOT")

    data_root = sys.argv[1]
    payload = json.loads(sys.stdin.read())
    peptides = payload.get("peptides", [])

    model = DeepImmuno(data_root)
    scores = model.score(peptides)

    sys.stdout.write(json.dumps({"scores": scores}))


if __name__ == "__main__":
    main()
