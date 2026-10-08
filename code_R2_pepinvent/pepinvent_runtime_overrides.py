#!/usr/bin/env python3
"""
Runtime-only PepINVENT overrides.

Nothing in this module writes to REINVENT source files.

When enabled via sitecustomize.py, it replaces methods only in the memory
of the current REINVENT Python process:

1) PepinventSampler.sample:
   - RL source pool may contain 6156 masked bladder peptides.
   - randomly select exactly sampler.batch_size source conditions when the
     source pool is larger than the requested batch;
   - temporarily set sampler.batch_size=1 so stock PepinventSampler generates
     exactly one completion per selected source instead of multiplying each
     source by batch_size.

2) TransformerModel.sample / likelihood for PepINVENT only:
   - generation is constrained token-by-token to CHUCKLES strings for the
     20 canonical amino acids ARNDCQEGHILKMFPSTWYV;
   - every filler uses the normal PepINVENT filler form used in the training
     logs, i.e. the canonical CHUCKLES amino-acid fragment with its final
     terminal atom removed;
   - the same constrained distribution is used for the Agent/Prior likelihood
     calculation so RL sampling and NLL calculation are consistent.

The original installed .py files are not edited, copied, patched, or rewritten.
"""

from __future__ import annotations

import importlib
import logging
import sys
import time
from pathlib import Path

import torch
from torch.autograd import Variable

from pepinvent_chuckles import AA_INTERNAL_CHUCKLES

_RUNTIME_MARKER = "PEPINVENT_RUNTIME_ONLY_NATURAL20_EXACT32_V2_NO_OPENBABEL"
_LOG = logging.getLogger("pepinvent_runtime_overrides")

_ORIGINAL_TRANSFORMER_SAMPLE = None
_ORIGINAL_TRANSFORMER_LIKELIHOOD = None
_ORIGINAL_PEPINVENT_SAMPLER_SAMPLE = None


def _stderr(msg):
    print(f"[{_RUNTIME_MARKER}] {msg}", file=sys.stderr, flush=True)


def _is_pepinvent_model(model) -> bool:
    identity = " ".join(
        [
            str(getattr(model, "_model_type", "")),
            model.__class__.__name__,
            model.__class__.__module__,
        ]
    ).lower()
    return "pepinvent" in identity or "pep_invent" in identity


def _find_pepinvent_sampler_class():
    import reinvent

    root = Path(reinvent.__file__).resolve().parent
    hits = []

    for path in root.rglob("*.py"):
        try:
            txt = path.read_text(encoding="utf-8")
        except Exception:
            continue
        if "class PepinventSampler" in txt:
            hits.append(path)

    if len(hits) != 1:
        raise RuntimeError(
            "Expected exactly one installed source file containing "
            f"class PepinventSampler; found {len(hits)}: {hits}"
        )

    path = hits[0]
    rel = path.relative_to(root).with_suffix("")
    module_name = "reinvent." + ".".join(rel.parts)
    module = importlib.import_module(module_name)

    if not hasattr(module, "PepinventSampler"):
        raise RuntimeError(
            f"{module_name} does not expose PepinventSampler"
        )

    return module.PepinventSampler


def _natural20_cache(model):
    cache = getattr(model, "_runtime_natural20_cache", None)
    if cache is not None:
        return cache

    # Fixed Natural20 internal CHUCKLES fragments.  No OpenBabel is used
    # at Python startup or during generation.
    aa_order = list("ARNDCQEGHILKMFPSTWYV")
    filler_strings = [
        AA_INTERNAL_CHUCKLES[aa]
        for aa in aa_order
    ]

    if len(set(filler_strings)) != 20:
        raise RuntimeError(
            "Natural20 CHUCKLES table did not produce 20 unique fillers."
        )

    def encode_filler(s):
        tokens = model.tokenizer.tokenize(
            s, with_begin_and_end=False
        )
        encoded_obj = model.vocabulary.encode(tokens)
        if hasattr(encoded_obj, "tolist"):
            encoded = [int(x) for x in encoded_obj.tolist()]
        else:
            encoded = [int(x) for x in encoded_obj]

        decoded = model.tokenizer.untokenize(
            model.vocabulary.decode(encoded)
        )
        if decoded != s:
            raise RuntimeError(
                "Natural20 filler failed vocabulary round trip: "
                f"{s} -> {decoded} | tokens={tokens} | ids={encoded}"
            )
        return tuple(encoded)

    filler_tokens = tuple(encode_filler(s) for s in filler_strings)

    cache = {
        "fillers": filler_tokens,
        "sep": int(model.vocabulary["|"]),
        "eos": int(model.vocabulary["$"]),
        "bos": int(model.vocabulary["^"]),
        "pad": int(getattr(model.vocabulary, "pad_token", 0)),
        "aa_order": aa_order,
        "filler_strings": filler_strings,
    }
    model._runtime_natural20_cache = cache

    _stderr(
        "Natural20 cache initialized: "
        "20 canonical amino-acid fillers, all using PepINVENT internal filler form."
    )
    return cache


def _source_spec(model, src_row):
    cache = _natural20_cache(model)

    ids = [int(x) for x in src_row.detach().cpu().tolist()]
    source = model.tokenizer.untokenize(
        model.vocabulary.decode(ids)
    )

    # Strip special-token artifacts if they are rendered by the tokenizer.
    parts = source.split("|")
    mask_count = sum(part == "?" for part in parts)

    if mask_count <= 0:
        return None

    return {
        "mask_count": mask_count,
        "source": source,
        "cache": cache,
    }


def _allowed_next(model, prefix_ids, spec):
    if spec is None:
        return None

    cache = spec["cache"]
    mask_count = spec["mask_count"]

    ids = [int(x) for x in prefix_ids]

    # After EOS, REINVENT's own break-condition/padding behavior takes over.
    if cache["eos"] in ids:
        return None

    # Remove BOS/PAD from the generated prefix.
    ids = [
        x for x in ids
        if x not in (cache["bos"], cache["pad"])
    ]

    sep = cache["sep"]
    slot = ids.count(sep)

    if slot >= mask_count:
        return [cache["eos"]]

    last_sep = -1
    for i, token_id in enumerate(ids):
        if token_id == sep:
            last_sep = i

    filler_prefix = tuple(ids[last_sep + 1:])
    boundary = cache["eos"] if slot == mask_count - 1 else sep

    allowed = set()
    for candidate in cache["fillers"]:
        n = len(filler_prefix)

        if n < len(candidate) and candidate[:n] == filler_prefix:
            allowed.add(candidate[n])
        elif n == len(candidate) and candidate == filler_prefix:
            allowed.add(boundary)

    if not allowed:
        decoded_prefix = model.tokenizer.untokenize(
            model.vocabulary.decode(ids)
        )
        raise RuntimeError(
            "Natural20 grammar reached an impossible generated prefix. "
            f"source={spec['source']} | prefix={decoded_prefix} | "
            f"slot={slot}/{mask_count}"
        )

    return sorted(allowed)


def _constrain_prob(model, prob, ys, src):
    constrained = prob.clone()

    for b in range(prob.shape[0]):
        spec = _source_spec(model, src[b])
        allowed = _allowed_next(
            model,
            ys[b].detach().cpu().tolist(),
            spec,
        )

        if allowed is None:
            continue

        allowed_idx = torch.tensor(
            allowed,
            dtype=torch.long,
            device=prob.device,
        )
        keep = torch.zeros(
            constrained[b].shape[0],
            dtype=torch.bool,
            device=prob.device,
        )
        keep[allowed_idx] = True

        row = constrained[b].masked_fill(~keep, 0.0)
        total = row.sum()

        if (
            not torch.isfinite(total)
            or float(total.detach().cpu()) <= 0.0
        ):
            raise RuntimeError(
                "Natural20 constrained probability mass is zero or non-finite. "
                f"Allowed token IDs={allowed}"
            )

        constrained[b] = row / total

    return constrained


def _constrain_log_prob(model, log_prob, src, trg):
    """
    log_prob: [batch, vocab, target_len]
    trg:      [batch, target_len], including BOS but excluding the final target
              because this follows TransformerModel.likelihood().
    """
    constrained = log_prob.clone()
    pad = int(getattr(model.vocabulary, "pad_token", 0))

    for b in range(log_prob.shape[0]):
        spec = _source_spec(model, src[b])

        for t in range(log_prob.shape[2]):
            prefix = trg[b, : t + 1].detach().cpu().tolist()

            # NLLLoss ignores PAD=0, so no constraint is needed after padding.
            if pad in [int(x) for x in prefix[1:]]:
                continue

            allowed = _allowed_next(model, prefix, spec)
            if allowed is None:
                continue

            allowed_idx = torch.tensor(
                allowed,
                dtype=torch.long,
                device=log_prob.device,
            )

            mask = torch.ones(
                log_prob.shape[1],
                dtype=torch.bool,
                device=log_prob.device,
            )
            mask[allowed_idx] = False

            row = constrained[b, :, t].masked_fill(
                mask, float("-inf")
            )
            log_z = torch.logsumexp(row, dim=0)

            if not torch.isfinite(log_z):
                raise RuntimeError(
                    "Natural20 likelihood normalization became non-finite."
                )

            constrained[b, :, t] = row - log_z

    return constrained


def _runtime_transformer_sample(self, src, src_mask, decode_type):
    """
    Runtime replacement for TransformerModel.sample().
    For non-PepINVENT models, delegate to the original method.
    """
    global _ORIGINAL_TRANSFORMER_SAMPLE

    if not _is_pepinvent_model(self):
        return _ORIGINAL_TRANSFORMER_SAMPLE(
            self, src, src_mask, decode_type
        )

    from reinvent.models.transformer.core.network.module.subsequent_mask import (
        subsequent_mask,
    )
    from reinvent.models.transformer.core.vocabulary import SMILESTokenizer

    if not self._sampling_modes_enum.is_supported_sampling_mode(
        decode_type
    ):
        raise ValueError(
            f"Sampling mode `{decode_type}` is not supported"
        )

    if decode_type == self._sampling_modes_enum.BEAMSEARCH:
        raise RuntimeError(
            "Natural20 runtime-only override currently supports "
            "PepINVENT multinomial/greedy decoding, not beam search."
        )

    batch_size = src.shape[0]
    ys = torch.ones(1).to(self.device)
    ys = (
        ys.repeat(batch_size, 1)
        .view(batch_size, 1)
        .type_as(src.data)
    )

    encoder_outputs = self.network.encode(src, src_mask)
    break_condition = torch.zeros(
        batch_size, dtype=torch.bool
    ).to(self.device)
    nlls = torch.zeros(batch_size).to(self.device)
    end_token = self.vocabulary["$"]

    for _ in range(self.max_sequence_length - 1):
        out = self.network.decode(
            encoder_outputs,
            src_mask,
            Variable(ys),
            Variable(
                subsequent_mask(ys.size(1)).type_as(src.data)
            ),
        )

        log_prob = self.network.generator(
            out[:, -1], self.temperature
        )
        prob = torch.exp(log_prob)

        mask_property_token = self.mask_property_tokens(
            batch_size
        )
        prob = prob.masked_fill(mask_property_token, 0)

        prob = _constrain_prob(self, prob, ys, src)
        log_prob = torch.log(
            prob.clamp_min(torch.finfo(prob.dtype).tiny)
        )

        if decode_type == self._sampling_modes_enum.GREEDY:
            _, next_word = torch.max(prob, dim=1)
            next_word = next_word.masked_fill(
                break_condition.to(self.device), 0
            )
            ys = torch.cat(
                [ys, next_word.unsqueeze(-1)], dim=1
            )
            nlls += self._nll_loss(log_prob, next_word)

        elif decode_type == self._sampling_modes_enum.MULTINOMIAL:
            next_word = torch.multinomial(prob, 1)
            break_t = torch.unsqueeze(
                break_condition, 1
            ).to(self.device)
            next_word = next_word.masked_fill(
                break_t, 0
            )
            ys = torch.cat([ys, next_word], dim=1)
            next_word = torch.reshape(
                next_word, (next_word.shape[0],)
            )
            nlls += self._nll_loss(log_prob, next_word)

        break_condition = (
            break_condition | (next_word == end_token)
        )

        if all(break_condition):
            break

    tokenizer = SMILESTokenizer()

    input_smiles_list = [
        tokenizer.untokenize(self.vocabulary.decode(seq))
        for seq in src.detach().cpu().numpy()
    ]
    output_smiles_list = [
        tokenizer.untokenize(self.vocabulary.decode(seq))
        for seq in ys.detach().cpu().numpy()
    ]
    nlls = nlls.detach().cpu().numpy()

    return input_smiles_list, output_smiles_list, nlls


def _runtime_transformer_likelihood(
    self, src, src_mask, trg, trg_mask
):
    global _ORIGINAL_TRANSFORMER_LIKELIHOOD

    if not _is_pepinvent_model(self):
        return _ORIGINAL_TRANSFORMER_LIKELIHOOD(
            self, src, src_mask, trg, trg_mask
        )

    trg_y = trg[:, 1:]
    trg_in = trg[:, :-1]

    out = self.network.forward(
        src, trg_in, src_mask, trg_mask
    )
    log_prob = self.network.generator(
        out, self.temperature
    ).transpose(1, 2)

    log_prob = _constrain_log_prob(
        self, log_prob, src, trg_in
    )

    nll = self._nll_loss(log_prob, trg_y).sum(dim=1)
    return nll


def _runtime_pepinvent_sampler_sample(self, smilies):
    """
    Runtime exact-batch behavior without editing PepinventSampler source.

    Training:
      6156 source conditions -> choose exactly sampler.batch_size (32) ->
      temporarily set batch_size=1 -> stock sampler emits 32 outputs total.

    Checkpoint evaluation / inference:
      if number of source conditions already equals requested batch_size,
      use all source rows exactly once.
    """
    global _ORIGINAL_PEPINVENT_SAMPLER_SAMPLE

    pool = list(smilies)
    requested = min(int(self.batch_size), len(pool))

    if requested <= 0:
        return _ORIGINAL_PEPINVENT_SAMPLER_SAMPLE(self, pool)

    if len(pool) > requested:
        idx = torch.randperm(len(pool))[:requested].tolist()
        selected = [pool[i] for i in idx]
    else:
        idx = list(range(len(pool)))
        selected = pool

    _LOG.info(
        "[RUNTIME-EXACT32] pool=%d | selected=%d | unique_indices=%d | "
        "original_sampler_batch_size=%d",
        len(pool),
        len(selected),
        len(set(idx)),
        int(self.batch_size),
    )

    t0 = time.perf_counter()
    old_batch_size = self.batch_size

    try:
        # Stock PepinventSampler contains:
        #     smilies = smilies * self.batch_size
        # Setting this to 1 gives exactly one completion per selected source.
        self.batch_size = 1
        result = _ORIGINAL_PEPINVENT_SAMPLER_SAMPLE(
            self, selected
        )
    finally:
        self.batch_size = old_batch_size

    generated_n = None
    for attr in ("smilies", "output", "items2", "nlls"):
        value = getattr(result, attr, None)
        try:
            generated_n = len(value)
            break
        except Exception:
            pass

    _LOG.info(
        "[RUNTIME-EXACT32] completed | generated=%s | elapsed=%.3fs",
        generated_n,
        time.perf_counter() - t0,
    )
    return result


def install():
    global _ORIGINAL_TRANSFORMER_SAMPLE
    global _ORIGINAL_TRANSFORMER_LIKELIHOOD
    global _ORIGINAL_PEPINVENT_SAMPLER_SAMPLE

    from reinvent.models.transformer.transformer import (
        TransformerModel,
    )

    if getattr(
        TransformerModel,
        "_pepinvent_runtime_only_override_installed",
        False,
    ):
        return

    PepinventSampler = _find_pepinvent_sampler_class()

    _ORIGINAL_TRANSFORMER_SAMPLE = TransformerModel.sample
    _ORIGINAL_TRANSFORMER_LIKELIHOOD = (
        TransformerModel.likelihood
    )
    _ORIGINAL_PEPINVENT_SAMPLER_SAMPLE = (
        PepinventSampler.sample
    )

    # In-memory replacement only. No source file is opened for writing.
    TransformerModel.sample = _runtime_transformer_sample
    TransformerModel.likelihood = (
        _runtime_transformer_likelihood
    )
    PepinventSampler.sample = (
        _runtime_pepinvent_sampler_sample
    )

    TransformerModel._pepinvent_runtime_only_override_installed = True
    PepinventSampler._pepinvent_runtime_only_override_installed = True

    _stderr(
        "Installed in-memory Natural20 + exact-batch overrides. "
        "No REINVENT .py file was modified."
    )
