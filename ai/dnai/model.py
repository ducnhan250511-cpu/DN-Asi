"""Causal Transformer (DN-AI Micro).

Architecture: token embedding, RoPE, pre-norm multi-head self-attention,
GELU feed-forward, RMSNorm, residual connections, tied output projection.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ai.dnai.tensor import (
    Tensor,
    add,
    add_causal_mask,
    apply_rope,
    embedding,
    gelu,
    matmul,
    merge_heads,
    mul,
    rmsnorm,
    scale,
    softmax_lastdim,
    split_heads,
    tensor,
)


@dataclass
class Arch:
    vocab_size: int
    d_model: int = 128
    n_layer: int = 4
    n_head: int = 4
    d_ff: int = 384
    max_seq: int = 96
    rope_theta: float = 10000.0

    @property
    def d_head(self) -> int:
        if self.d_model % self.n_head != 0:
            raise ValueError("d_model must divide n_head")
        return self.d_model // self.n_head

    def to_dict(self) -> dict:
        return {
            "vocab_size": self.vocab_size,
            "d_model": self.d_model,
            "n_layer": self.n_layer,
            "n_head": self.n_head,
            "d_ff": self.d_ff,
            "max_seq": self.max_seq,
            "rope_theta": self.rope_theta,
        }

    @staticmethod
    def from_dict(d: dict) -> "Arch":
        return Arch(
            vocab_size=int(d["vocab_size"]),
            d_model=int(d["d_model"]),
            n_layer=int(d["n_layer"]),
            n_head=int(d["n_head"]),
            d_ff=int(d["d_ff"]),
            max_seq=int(d["max_seq"]),
            rope_theta=float(d.get("rope_theta", 10000.0)),
        )


def init_params(arch: Arch, rng: np.random.Generator) -> dict[str, Tensor]:
    def w(shape: tuple[int, ...]) -> Tensor:
        arr = rng.normal(0, 0.02, size=shape).astype(np.float32)
        return Tensor(arr)

    params: dict[str, Tensor] = {
        "embed": w((arch.vocab_size, arch.d_model)),
        "final_norm": Tensor(np.ones(arch.d_model, dtype=np.float32)),
    }
    for i in range(arch.n_layer):
        p = f"{i}."
        params[p + "norm1"] = Tensor(np.ones(arch.d_model, dtype=np.float32))
        params[p + "wq"] = w((arch.d_model, arch.d_model))
        params[p + "wk"] = w((arch.d_model, arch.d_model))
        params[p + "wv"] = w((arch.d_model, arch.d_model))
        params[p + "wo"] = w((arch.d_model, arch.d_model))
        params[p + "norm2"] = Tensor(np.ones(arch.d_model, dtype=np.float32))
        params[p + "w1"] = w((arch.d_model, arch.d_ff))
        params[p + "w2"] = w((arch.d_ff, arch.d_model))
    return params


def rope_cache(arch: Arch, seq: int) -> tuple[np.ndarray, np.ndarray]:
    half = arch.d_head // 2
    pos = np.arange(seq, dtype=np.float32)
    inv = np.float32(1.0) / (
        np.float32(arch.rope_theta) ** (np.arange(half, dtype=np.float32) / np.float32(half))
    )
    angles = pos[:, None] * inv[None, :]
    return np.cos(angles).astype(np.float32), np.sin(angles).astype(np.float32)


def forward_logits(ids: np.ndarray, params: dict[str, Tensor], arch: Arch) -> Tensor:
    """ids: (B, T) int64. Returns logits (B, T, V)."""
    b, t = ids.shape
    if t > arch.max_seq:
        raise ValueError(f"sequence {t} exceeds max_seq {arch.max_seq}")
    cos, sin = rope_cache(arch, t)
    x = embedding(ids, params["embed"])
    for i in range(arch.n_layer):
        p = f"{i}."
        h = rmsnorm(x, params[p + "norm1"])
        q = split_heads(matmul(h, params[p + "wq"]), arch.n_head)
        k = split_heads(matmul(h, params[p + "wk"]), arch.n_head)
        v = split_heads(matmul(h, params[p + "wv"]), arch.n_head)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        scale_v = np.float32(1.0 / math.sqrt(arch.d_head))
        scores = scale(matmul(q, _transpose_time(k)), float(scale_v))
        scores = add_causal_mask(scores)
        probs = softmax_lastdim(scores)
        attn = merge_heads(matmul(probs, v))
        x = add(x, matmul(attn, params[p + "wo"]))
        h2 = rmsnorm(x, params[p + "norm2"])
        hidden = gelu(matmul(h2, params[p + "w1"]))
        x = add(x, matmul(hidden, params[p + "w2"]))
    x = rmsnorm(x, params["final_norm"])
    # Tied embedding: logits = x @ E^T
    return matmul(x, _transpose_2d(params["embed"]))


def _transpose_time(k: Tensor) -> Tensor:
    data = np.swapaxes(k.data, -1, -2)
    out = Tensor(np.ascontiguousarray(data))

    def bw() -> None:
        if k.requires_grad and k.grad is not None:
            k.grad += np.swapaxes(out.grad, -1, -2)

    out._prev = [k] if k.requires_grad else []
    out._bw = bw if out._prev else None
    return out


def _transpose_2d(w: Tensor) -> Tensor:
    data = w.data.T.copy()
    out = Tensor(data)

    def bw() -> None:
        if w.requires_grad and w.grad is not None:
            w.grad += out.grad.T

    out._prev = [w] if w.requires_grad else []
    out._bw = bw if out._prev else None
    return out


def loss_from_logits(logits: Tensor, labels: np.ndarray, mask: np.ndarray) -> Tensor:
    """labels/mask: (B, T). mask 1 = contribute. Returns scalar mean loss."""
    raw = logits.data
    m = np.max(raw, axis=-1, keepdims=True)
    e = np.exp(raw - m)
    probs = e / np.sum(e, axis=-1, keepdims=True)
    b, t = labels.shape
    nll = -np.log(probs[np.arange(b)[:, None], np.arange(t)[None, :], labels] + np.float32(1e-9))
    weight = mask.astype(np.float32)
    denom = np.float32(max(float(weight.sum()), 1.0))
    loss_val = np.float32(np.sum(nll * weight) / denom)
    out = Tensor(np.array(loss_val, dtype=np.float32))

    def bw() -> None:
        g = probs.copy()
        g[np.arange(b)[:, None], np.arange(t)[None, :], labels] -= np.float32(1.0)
        g *= weight[..., None] / denom
        g *= out.grad.reshape(())
        logits.grad += g

    out._prev = [logits]
    out._bw = bw
    return out


def numpy_params(params: dict[str, Tensor]) -> dict[str, np.ndarray]:
    return {k: v.data.astype(np.float32) for k, v in params.items()}


def load_params(arrays: dict[str, np.ndarray]) -> dict[str, Tensor]:
    return {k: Tensor(v.astype(np.float32)) for k, v in arrays.items()}


def zero_grads(params: dict[str, Tensor]) -> None:
    for p in params.values():
        p.zero_grad()


def forward_numpy(ids: np.ndarray, arrays: dict[str, np.ndarray], arch: Arch) -> np.ndarray:
    """Inference forward, no graph. ids (T,) -> logits (T, V)."""
    return _forward_numpy_batch(ids[None, :], arrays, arch)[0]


def _forward_numpy_batch(ids: np.ndarray, arrays: dict[str, np.ndarray], arch: Arch) -> np.ndarray:
    _, t = ids.shape
    cos, sin = rope_cache(arch, t)
    x = arrays["embed"][ids]
    for i in range(arch.n_layer):
        p = f"{i}."
        h = _rms(x, arrays[p + "norm1"])
        q = _heads(_lin(h, arrays[p + "wq"]), arch.n_head)
        k = _heads(_lin(h, arrays[p + "wk"]), arch.n_head)
        v = _heads(_lin(h, arrays[p + "wv"]), arch.n_head)
        q = _rope_np(q, cos, sin)
        k = _rope_np(k, cos, sin)
        scale_v = np.float32(1.0 / math.sqrt(arch.d_head))
        scores = np.matmul(q, np.swapaxes(k, -1, -2)) * scale_v
        mask = np.triu(np.full((t, t), np.float32(-1e9)), k=1)
        scores = scores + mask
        scores = scores - scores.max(axis=-1, keepdims=True)
        probs = np.exp(scores)
        probs = probs / probs.sum(axis=-1, keepdims=True)
        attn = np.matmul(probs, v)
        b, heads, tt, dh = attn.shape
        attn = np.ascontiguousarray(attn.transpose(0, 2, 1, 3)).reshape(b, tt, heads * dh)
        x = x + _lin(attn, arrays[p + "wo"])
        h2 = _rms(x, arrays[p + "norm2"])
        hidden = _gelu_np(_lin(h2, arrays[p + "w1"]))
        x = x + _lin(hidden, arrays[p + "w2"])
    x = _rms(x, arrays["final_norm"])
    return x @ arrays["embed"].T


def _lin(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    return x @ w


def _rms(x: np.ndarray, w: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    ms = np.mean(x * x, axis=-1, keepdims=True)
    return x / np.sqrt(ms + np.float32(eps)) * w


def _heads(x: np.ndarray, n_head: int) -> np.ndarray:
    b, t, d = x.shape
    dh = d // n_head
    return np.ascontiguousarray(x.reshape(b, t, n_head, dh).transpose(0, 2, 1, 3))


def _rope_np(x: np.ndarray, cos: np.ndarray, sin: np.ndarray) -> np.ndarray:
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    c = cos.reshape(1, 1, cos.shape[0], half)
    s = sin.reshape(1, 1, sin.shape[0], half)
    return np.concatenate([x1 * c - x2 * s, x1 * s + x2 * c], axis=-1)


def _gelu_np(x: np.ndarray) -> np.ndarray:
    k = np.float32(0.7978845608028654)
    u = k * (x + np.float32(0.044715) * x * x * x)
    return np.float32(0.5) * x * (np.float32(1.0) + np.tanh(u))


def gradcheck(seed: int = 0) -> None:
    """Finite-difference check on a tiny network."""
    rng = np.random.default_rng(seed)
    arch = Arch(vocab_size=20, d_model=8, n_layer=1, n_head=2, d_ff=16, max_seq=6)
    params = init_params(arch, rng)
    ids = rng.integers(0, 20, size=(2, 5))
    labels = np.concatenate([ids[:, 1:], rng.integers(0, 20, size=(2, 1))], axis=1)
    mask = np.ones_like(labels, dtype=np.float32)
    zero_grads(params)
    logits = forward_logits(ids, params, arch)
    loss = loss_from_logits(logits, labels, mask)
    loss.backward()
    key = "0.w1"
    analytic = params[key].grad.copy()
    eps = 1e-3
    flat = params[key].data.reshape(-1)
    num = np.zeros_like(flat)
    # Check 6 random coordinates.
    picks = rng.choice(flat.size, size=6, replace=False)
    for i in picks:
        orig = flat[i]
        flat[i] = orig + eps
        lp = loss_from_logits(forward_logits(ids, params, arch), labels, mask).data
        flat[i] = orig - eps
        lm = loss_from_logits(forward_logits(ids, params, arch), labels, mask).data
        flat[i] = orig
        num[i] = (lp - lm) / (2 * eps)
    a = analytic.reshape(-1)[picks]
    n = num[picks]
    if not np.allclose(a, n, rtol=1e-2, atol=1e-2):
        raise SystemExit(f"gradcheck failed\nanalytic {a}\nnumeric {n}")
    print("gradcheck ok", a[:3], n[:3])


if __name__ == "__main__":
    gradcheck()
