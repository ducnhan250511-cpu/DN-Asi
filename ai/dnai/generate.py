"""Autoregressive generation: temperature, top-k, top-p, repetition penalty."""

from __future__ import annotations

import numpy as np

from ai.dnai.model import Arch, _forward_numpy_batch


def softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - np.max(logits)
    e = np.exp(z)
    return e / np.sum(e)


def sample_id(
    logits: np.ndarray,
    rng: np.random.Generator,
    temperature: float,
    top_k: int,
    top_p: float,
    recent: list[int],
    repetition_penalty: float = 1.12,
) -> tuple[int, float]:
    x = logits.astype(np.float32).copy()
    if repetition_penalty != 1.0 and recent:
        for tid in set(recent[-24:]):
            if x[tid] > 0:
                x[tid] /= np.float32(repetition_penalty)
            else:
                x[tid] *= np.float32(repetition_penalty)
    temp = max(float(temperature), 1e-4)
    x = x / np.float32(temp)
    if top_k and top_k < x.shape[0]:
        cut = np.partition(x, -top_k)[-top_k]
        x = np.where(x < cut, np.float32(-1e9), x)
    order = np.argsort(x)[::-1]
    probs_sorted = softmax(x[order])
    if 0 < top_p < 1:
        cum = np.cumsum(probs_sorted)
        keep = max(1, int(np.searchsorted(cum, top_p) + 1))
        banned = order[keep:]
        x[banned] = np.float32(-1e9)
    probs = softmax(x)
    if temperature < 1e-3:
        idx = int(np.argmax(probs))
    else:
        idx = int(rng.choice(probs.shape[0], p=probs))
    return idx, float(np.log(probs[idx] + 1e-9))


def generate(
    prompt: list[int],
    arrays: dict[str, np.ndarray],
    arch: Arch,
    eos_id: int,
    max_new: int,
    temperature: float,
    top_k: int,
    top_p: float,
    seed: int,
) -> tuple[list[int], float]:
    rng = np.random.default_rng(seed)
    ids = list(prompt)
    logprobs: list[float] = []
    for _ in range(max_new):
        window = ids[-arch.max_seq :]
        logits = _forward_numpy_batch(np.asarray(window, dtype=np.int64)[None, :], arrays, arch)
        nxt, lp = sample_id(logits[0, -1], rng, temperature, top_k, top_p, ids)
        ids.append(nxt)
        logprobs.append(lp)
        if nxt == eos_id:
            break
    return ids[len(prompt) :], (sum(logprobs) / len(logprobs) if logprobs else 0.0)
