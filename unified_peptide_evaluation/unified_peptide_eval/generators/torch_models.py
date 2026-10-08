from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils import AA_ORDER, set_seed, unwrap_state_dict


class ResBlock(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.res_block = nn.Sequential(
            nn.ReLU(True),
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=1),
        )

    def forward(self, x):
        return x + 0.3 * self.res_block(x)


class GeneratorGAN(nn.Module):
    def __init__(self, hidden, seq_len, n_chars):
        super().__init__()
        self.fc1 = nn.Linear(128, hidden * seq_len)
        self.block = nn.Sequential(*[ResBlock(hidden) for _ in range(5)])
        self.conv1 = nn.Conv1d(hidden, n_chars, kernel_size=1)
        self.hidden, self.seq_len, self.n_chars = hidden, seq_len, n_chars

    def forward(self, noise):
        b = noise.shape[0]
        x = self.fc1(noise).view(b, self.hidden, self.seq_len)
        x = self.block(x)
        x = self.conv1(x).transpose(1, 2).contiguous()
        x = x.view(b * self.seq_len, self.n_chars)
        x = F.gumbel_softmax(x, tau=0.75, hard=False)
        return x.view(b, self.seq_len, self.n_chars)


class GeneratorLSTM(nn.Module):
    def __init__(self, hidden, seq_len, n_chars, num_layers=4, bidirectional=False):
        super().__init__()
        self.fc1 = nn.Linear(128, hidden * seq_len)
        self.lstm = nn.LSTM(
            input_size=hidden,
            hidden_size=hidden,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
        )
        out_dim = hidden * (2 if bidirectional else 1)
        self.proj = nn.Linear(out_dim, n_chars)
        self.hidden, self.seq_len, self.n_chars = hidden, seq_len, n_chars

    def forward(self, noise):
        b = noise.shape[0]
        x = self.fc1(noise).view(b, self.seq_len, self.hidden)
        x, _ = self.lstm(x)
        x = self.proj(x).contiguous().view(b * self.seq_len, self.n_chars)
        x = F.gumbel_softmax(x, tau=0.75, hard=False)
        return x.view(b, self.seq_len, self.n_chars)


def build_rotary_embeddings(seq_len, head_dim, device, base=10000):
    pos = torch.arange(seq_len, dtype=torch.float32, device=device)
    freqs = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    angles = torch.outer(pos, freqs)
    return torch.cos(angles)[None, None, ...], torch.sin(angles)[None, None, ...]


def apply_rotary(x, cos, sin):
    x1, x2 = x[..., ::2], x[..., 1::2]
    return torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1).flatten(-2)


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        norm = x.norm(dim=-1, keepdim=True)
        return self.weight * x / (norm / (x.size(-1) ** 0.5) + self.eps)


class GQAAttention(nn.Module):
    def __init__(self, embed_dim, q_heads, kv_heads):
        super().__init__()
        if embed_dim % q_heads != 0:
            raise ValueError("embed_dim must be divisible by q_heads")
        if q_heads % kv_heads != 0:
            raise ValueError("q_heads must be divisible by kv_heads")
        self.q_heads, self.kv_heads = q_heads, kv_heads
        self.head_dim = embed_dim // q_heads
        if self.head_dim % 2:
            raise ValueError("RoPE requires even head_dim")
        self.scale = self.head_dim ** 0.5
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, self.head_dim * kv_heads)
        self.v_proj = nn.Linear(embed_dim, self.head_dim * kv_heads)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x, cos, sin):
        b, t, _ = x.shape
        q = self.q_proj(x).view(b, t, self.q_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, t, self.kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, t, self.kv_heads, self.head_dim).transpose(1, 2)
        k = k.repeat_interleave(self.q_heads // self.kv_heads, dim=1)
        v = v.repeat_interleave(self.q_heads // self.kv_heads, dim=1)
        q = apply_rotary(q, cos, sin)
        k = apply_rotary(k, cos, sin)
        weights = F.softmax((q @ k.transpose(-2, -1)) / self.scale, dim=-1)
        out = weights @ v
        return self.out_proj(out.transpose(1, 2).reshape(b, t, -1))


class FFNLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.ffn = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim))

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
        head_dim = self.embed_dim // self.q_heads
        cos, sin = build_rotary_embeddings(x.size(1), head_dim, x.device)
        x = self.attn(x, cos, sin) + x
        x = self.rms1(x)
        x = self.ffn(x) + x
        return self.rms2(x)


class GeneratorTransformer(nn.Module):
    def __init__(self, hidden, seq_len, n_chars, num_layers=2, q_heads=8, kv_heads=8):
        super().__init__()
        self.fc1 = nn.Linear(128, hidden * seq_len)
        self.layers = nn.ModuleList([TransformerBlock(hidden, q_heads, kv_heads) for _ in range(num_layers)])
        self.proj = nn.Linear(hidden, n_chars)
        self.hidden, self.seq_len, self.n_chars = hidden, seq_len, n_chars

    def forward(self, noise):
        b = noise.shape[0]
        x = self.fc1(noise).view(b, self.seq_len, self.hidden)
        for layer in self.layers:
            x = layer(x)
        x = self.proj(x).contiguous().view(b * self.seq_len, self.n_chars)
        x = F.gumbel_softmax(x, tau=0.75, hard=False)
        return x.view(b, self.seq_len, self.n_chars)


class GeneratorD3PM(nn.Module):
    def __init__(self, hidden, seq_len, n_chars, num_diffusion_steps, num_layers=2, q_heads=8, kv_heads=8):
        super().__init__()
        self.seq_len, self.n_chars, self.hidden = seq_len, n_chars, hidden
        self.input_proj = nn.Linear(n_chars, hidden)
        self.time_embedding = nn.Embedding(num_diffusion_steps, hidden)
        self.layers = nn.ModuleList([TransformerBlock(hidden, q_heads, kv_heads) for _ in range(num_layers)])
        self.proj = nn.Linear(hidden, n_chars)

    def forward(self, noisy_data, t):
        h = self.input_proj(noisy_data) + self.time_embedding(t).unsqueeze(1)
        for layer in self.layers:
            h = layer(h)
        return self.proj(h)


class D3PMUniform:
    def __init__(self, num_steps, n_chars, device):
        self.num_steps, self.n_chars, self.device = num_steps, n_chars, device
        self.eps = 1e-8
        steps = torch.arange(num_steps + 1, dtype=torch.float64, device=device) / num_steps
        alpha_bar = torch.cos((steps + 0.008) / 1.008 * np.pi / 2)
        betas = torch.clamp(1.0 - alpha_bar[1:] / alpha_bar[:-1], max=0.999)
        self.betas = betas.float()
        eye = torch.eye(n_chars, dtype=torch.float64, device=device)
        uniform = torch.ones((n_chars, n_chars), dtype=torch.float64, device=device) / n_chars
        q_one_step = [(1.0 - b) * eye + b * uniform for b in betas]
        self.Q = torch.stack(q_one_step, dim=0).float()
        qbar, running = [], eye.clone()
        for q in q_one_step:
            running = running @ q
            qbar.append(running.clone())
        self.Qbar = torch.stack(qbar, dim=0).float()
        self.Qbar_prev = torch.cat([torch.eye(n_chars, device=device)[None, ...], self.Qbar[:-1]], dim=0)

    def posterior_from_start_probs(self, start_probs, x_t_idx, t):
        b, l, k = start_probs.shape
        out = torch.empty_like(start_probs)
        zero_mask = t == 0
        if zero_mask.any():
            out[zero_mask] = start_probs[zero_mask]
        nz = ~zero_mask
        if nz.any():
            probs0 = start_probs[nz]
            xt = x_t_idx[nz]
            tnz = t[nz]
            Qt = self.Q[tnz]
            Qbar_prev = self.Qbar_prev[tnz]
            xt_expand = xt.unsqueeze(1).expand(-1, k, -1)
            fact1 = torch.gather(Qt, 2, xt_expand).transpose(1, 2)
            fact2 = torch.einsum("blk,bkj->blj", probs0, Qbar_prev)
            posterior = fact1 * fact2
            out[nz] = posterior / posterior.sum(dim=-1, keepdim=True).clamp_min(self.eps)
        return out

    @torch.inference_mode()
    def sample(self, model, batch_size, seq_len):
        x = torch.randint(0, self.n_chars, (batch_size, seq_len), device=self.device)
        for step in range(self.num_steps - 1, -1, -1):
            t = torch.full((batch_size,), step, device=self.device, dtype=torch.long)
            x_onehot = F.one_hot(x, num_classes=self.n_chars).float()
            pred_x0_probs = torch.softmax(model(x_onehot, t), dim=-1)
            reverse_probs = self.posterior_from_start_probs(pred_x0_probs, x, t)
            if step == 0:
                x = torch.argmax(reverse_probs, dim=-1)
            else:
                x = torch.multinomial(reverse_probs.reshape(-1, self.n_chars), 1).view(batch_size, seq_len)
        return x


def detect_architecture(state_dict) -> str:
    keys = set(state_dict)
    if any(k.startswith("time_embedding.") for k in keys) and any(k.startswith("input_proj.") for k in keys):
        return "d3pm"
    if any(k.startswith("lstm.") for k in keys):
        return "lstm"
    if any(k.startswith("layers.") and ".attn." in k for k in keys) and "fc1.weight" in keys:
        return "transformer"
    if "conv1.weight" in keys and "fc1.weight" in keys:
        return "gan"
    raise RuntimeError("Could not infer architecture from checkpoint keys. Use --architecture explicitly.")


def _num_layers_from_keys(state_dict, prefix="layers"):
    vals = []
    rgx = re.compile(rf"^{re.escape(prefix)}\.(\d+)\.")
    for k in state_dict:
        m = rgx.match(k)
        if m:
            vals.append(int(m.group(1)))
    return max(vals) + 1 if vals else 0


def _lstm_layers(state_dict):
    vals = []
    for k in state_dict:
        m = re.match(r"lstm\.weight_ih_l(\d+)", k)
        if m:
            vals.append(int(m.group(1)))
    return max(vals) + 1 if vals else 1


def build_model_from_state(state_dict, architecture="auto", q_heads=8, kv_heads=8):
    arch = detect_architecture(state_dict) if architecture == "auto" else architecture.lower()
    if arch == "gan":
        n_chars, hidden = state_dict["conv1.weight"].shape[:2]
        seq_len = state_dict["fc1.weight"].shape[0] // hidden
        model = GeneratorGAN(hidden, seq_len, n_chars)
        meta = dict(hidden=hidden, seq_len=seq_len, n_chars=n_chars)
    elif arch == "lstm":
        n_chars = state_dict["proj.weight"].shape[0]
        out_dim = state_dict["proj.weight"].shape[1]
        hidden = state_dict["lstm.weight_hh_l0"].shape[1]
        bidirectional = any("_reverse" in k for k in state_dict)
        seq_len = state_dict["fc1.weight"].shape[0] // hidden
        num_layers = _lstm_layers(state_dict)
        model = GeneratorLSTM(hidden, seq_len, n_chars, num_layers, bidirectional)
        meta = dict(hidden=hidden, seq_len=seq_len, n_chars=n_chars, num_layers=num_layers, bidirectional=bidirectional)
    elif arch == "transformer":
        n_chars, hidden = state_dict["proj.weight"].shape
        seq_len = state_dict["fc1.weight"].shape[0] // hidden
        num_layers = _num_layers_from_keys(state_dict)
        # q_heads is not recoverable from a dense q_proj state_dict. Default matches uploaded training code.
        k_out = state_dict["layers.0.attn.k_proj.weight"].shape[0]
        head_dim = hidden // q_heads
        inferred_kv = k_out // head_dim
        if inferred_kv > 0:
            kv_heads = inferred_kv
        model = GeneratorTransformer(hidden, seq_len, n_chars, num_layers, q_heads, kv_heads)
        meta = dict(hidden=hidden, seq_len=seq_len, n_chars=n_chars, num_layers=num_layers, q_heads=q_heads, kv_heads=kv_heads)
    elif arch in {"d3pm", "diffusion"}:
        n_chars, hidden = state_dict["proj.weight"].shape
        seq_len = 10  # The uploaded D3PM architecture is fixed at peptide length 10 incl. placeholder.
        num_steps = state_dict["time_embedding.weight"].shape[0]
        num_layers = _num_layers_from_keys(state_dict)
        k_out = state_dict["layers.0.attn.k_proj.weight"].shape[0]
        head_dim = hidden // q_heads
        inferred_kv = k_out // head_dim
        if inferred_kv > 0:
            kv_heads = inferred_kv
        model = GeneratorD3PM(hidden, seq_len, n_chars, num_steps, num_layers, q_heads, kv_heads)
        meta = dict(hidden=hidden, seq_len=seq_len, n_chars=n_chars, num_diffusion_steps=num_steps,
                    num_layers=num_layers, q_heads=q_heads, kv_heads=kv_heads)
        arch = "d3pm"
    else:
        raise ValueError(f"Unsupported architecture: {architecture}")
    model.load_state_dict(state_dict, strict=True)
    return model, arch, meta


def indices_to_peptides(idx: torch.Tensor):
    arr = idx.detach().cpu().numpy()
    return ["".join(AA_ORDER[int(i)] for i in row) for row in arr]


def generate_from_pytorch_checkpoint(
    checkpoint: str | Path,
    architecture: str,
    num_samples: int,
    seed: int,
    batch_size: int,
    device: str,
    q_heads: int = 8,
    kv_heads: int = 8,
):
    set_seed(seed)
    dev = torch.device(device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    raw = torch.load(checkpoint, map_location="cpu")
    state = unwrap_state_dict(raw)
    model, arch, meta = build_model_from_state(state, architecture, q_heads=q_heads, kv_heads=kv_heads)
    model = model.to(dev).eval()

    peptides = []
    if arch == "d3pm":
        batch_size = min(int(batch_size), 64)
        diffusion = D3PMUniform(meta["num_diffusion_steps"], meta["n_chars"], dev)
        while len(peptides) < num_samples:
            n = min(batch_size, num_samples - len(peptides))
            idx = diffusion.sample(model, n, meta["seq_len"])
            peptides.extend(indices_to_peptides(idx))
    else:
        with torch.inference_mode():
            while len(peptides) < num_samples:
                n = min(batch_size, num_samples - len(peptides))
                noise = torch.randn(n, 128, device=dev)
                probs = model(noise)
                idx = torch.argmax(probs, dim=2)
                peptides.extend(indices_to_peptides(idx))
    return peptides, arch, meta
