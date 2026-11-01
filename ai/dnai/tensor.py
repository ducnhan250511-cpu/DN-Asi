"""Minimal reverse-mode autodiff for float32 tensors (NumPy only)."""

from __future__ import annotations

from typing import Callable

import numpy as np

_REDUCE = Callable[[], None]


def _sum_to(grad: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    for axis, (g, s) in enumerate(zip(grad.shape, shape)):
        if s == 1 and g != 1:
            grad = grad.sum(axis=axis, keepdims=True)
    return grad


class Tensor:
    def __init__(self, data: np.ndarray, requires_grad: bool = True):
        self.data = np.asarray(data, dtype=np.float32)
        self.requires_grad = requires_grad
        self.grad = np.zeros_like(self.data) if requires_grad else None
        self._prev: list[Tensor] = []
        self._bw: _REDUCE | None = None

    @property
    def shape(self) -> tuple[int, ...]:
        return self.data.shape

    def zero_grad(self) -> None:
        if self.requires_grad and self.grad is not None:
            self.grad.fill(0.0)

    def backward(self) -> None:
        topo: list[Tensor] = []
        seen: set[int] = set()

        def build(node: Tensor) -> None:
            if id(node) in seen:
                return
            seen.add(id(node))
            for parent in node._prev:
                build(parent)
            topo.append(node)

        build(self)
        self.grad = np.ones_like(self.data)
        for node in reversed(topo):
            if node._bw is not None:
                node._bw()


def tensor(data, requires_grad: bool = True) -> Tensor:
    return Tensor(np.asarray(data, dtype=np.float32), requires_grad=requires_grad)


def _link(out: Tensor, parents: list[Tensor], bw: _REDUCE) -> Tensor:
    out._prev = [p for p in parents if p.requires_grad]
    out._bw = bw if out._prev else None
    if not out._prev:
        out.requires_grad = False
        out.grad = None
    return out


def matmul(a: Tensor, b: Tensor) -> Tensor:
    out = Tensor(a.data @ b.data)

    def bw() -> None:
        g = out.grad
        if a.requires_grad and a.grad is not None:
            a.grad += _sum_to(g @ np.swapaxes(b.data, -1, -2), a.shape)
        if b.requires_grad and b.grad is not None:
            b.grad += _sum_to(np.swapaxes(a.data, -1, -2) @ g, b.shape)

    return _link(out, [a, b], bw)


def add(a: Tensor, b: Tensor) -> Tensor:
    out = Tensor(a.data + b.data)

    def bw() -> None:
        if a.requires_grad and a.grad is not None:
            a.grad += _sum_to(out.grad, a.shape)
        if b.requires_grad and b.grad is not None:
            b.grad += _sum_to(out.grad, b.shape)

    return _link(out, [a, b], bw)


def mul(a: Tensor, b: Tensor) -> Tensor:
    out = Tensor(a.data * b.data)

    def bw() -> None:
        if a.requires_grad and a.grad is not None:
            a.grad += _sum_to(out.grad * b.data, a.shape)
        if b.requires_grad and b.grad is not None:
            b.grad += _sum_to(out.grad * a.data, b.shape)

    return _link(out, [a, b], bw)


def scale(a: Tensor, c: float) -> Tensor:
    out = Tensor(a.data * np.float32(c))

    def bw() -> None:
        if a.requires_grad and a.grad is not None:
            a.grad += out.grad * np.float32(c)

    return _link(out, [a], bw)


def gelu(a: Tensor) -> Tensor:
    """tanh approximation of GELU. Must match the TypeScript runtime."""
    x = a.data
    k = np.float32(0.7978845608028654)  # sqrt(2/pi)
    u = k * (x + np.float32(0.044715) * x * x * x)
    t = np.tanh(u)
    out = Tensor(np.float32(0.5) * x * (np.float32(1.0) + t))

    def bw() -> None:
        if not a.requires_grad or a.grad is None:
            return
        dinner = k * (np.float32(1.0) + np.float32(0.134145) * x * x)
        dy_dx = np.float32(0.5) * (np.float32(1.0) + t) + np.float32(0.5) * x * (
            np.float32(1.0) - t * t
        ) * dinner
        a.grad += out.grad * dy_dx

    return _link(out, [a], bw)


def rmsnorm(x: Tensor, weight: Tensor, eps: float = 1e-5) -> Tensor:
    ms = np.mean(x.data * x.data, axis=-1, keepdims=True)
    inv = np.reciprocal(np.sqrt(ms + np.float32(eps)))
    normed = x.data * inv
    out = Tensor(normed * weight.data)

    def bw() -> None:
        g = out.grad
        if weight.requires_grad and weight.grad is not None:
            weight.grad += _sum_to(g * normed, weight.shape)
        if x.requires_grad and x.grad is not None:
            gn = g * weight.data
            mean_term = np.mean(gn * normed, axis=-1, keepdims=True)
            x.grad += inv * (gn - normed * mean_term)

    return _link(out, [x, weight], bw)


def softmax_lastdim(a: Tensor) -> Tensor:
    m = np.max(a.data, axis=-1, keepdims=True)
    e = np.exp(a.data - m)
    out_data = e / np.sum(e, axis=-1, keepdims=True)
    out = Tensor(out_data)

    def bw() -> None:
        if not a.requires_grad or a.grad is None:
            return
        g = out.grad
        dot = np.sum(g * out_data, axis=-1, keepdims=True)
        a.grad += out_data * (g - dot)

    return _link(out, [a], bw)


def embedding(ids: np.ndarray, table: Tensor) -> Tensor:
    out = Tensor(table.data[ids])

    def bw() -> None:
        if table.requires_grad and table.grad is not None:
            np.add.at(table.grad, ids, out.grad)

    return _link(out, [table], bw)


def split_heads(x: Tensor, n_head: int) -> Tensor:
    b, t, d = x.data.shape
    dh = d // n_head
    data = np.ascontiguousarray(x.data.reshape(b, t, n_head, dh).transpose(0, 2, 1, 3))
    out = Tensor(data)

    def bw() -> None:
        if x.requires_grad and x.grad is not None:
            x.grad += np.ascontiguousarray(out.grad.transpose(0, 2, 1, 3)).reshape(b, t, d)

    return _link(out, [x], bw)


def merge_heads(x: Tensor) -> Tensor:
    b, h, t, dh = x.data.shape
    data = np.ascontiguousarray(x.data.transpose(0, 2, 1, 3)).reshape(b, t, h * dh)
    out = Tensor(data)

    def bw() -> None:
        if x.requires_grad and x.grad is not None:
            back = out.grad.reshape(b, t, h, dh).transpose(0, 2, 1, 3)
            x.grad += np.ascontiguousarray(back)

    return _link(out, [x], bw)


def add_causal_mask(scores: Tensor) -> Tensor:
    t = scores.data.shape[-1]
    mask = np.triu(np.full((t, t), np.float32(-1e9), dtype=np.float32), k=1)
    out = Tensor(scores.data + mask)

    def bw() -> None:
        if scores.requires_grad and scores.grad is not None:
            scores.grad += out.grad

    return _link(out, [scores], bw)


def apply_rope(x: Tensor, cos: np.ndarray, sin: np.ndarray) -> Tensor:
    """GPT-NeoX style half-split RoPE. cos/sin: (T, Dh/2)."""
    half = x.data.shape[-1] // 2
    x1 = x.data[..., :half]
    x2 = x.data[..., half:]
    # cos/sin broadcast over (B, H, T, half)
    c = cos.reshape(1, 1, cos.shape[0], half)
    s = sin.reshape(1, 1, sin.shape[0], half)
    data = np.concatenate([x1 * c - x2 * s, x1 * s + x2 * c], axis=-1)
    out = Tensor(data)

    def bw() -> None:
        if not x.requires_grad or x.grad is None:
            return
        g1 = out.grad[..., :half]
        g2 = out.grad[..., half:]
        dx1 = g1 * c + g2 * s
        dx2 = -g1 * s + g2 * c
        x.grad += np.concatenate([dx1, dx2], axis=-1)

    return _link(out, [x], bw)
