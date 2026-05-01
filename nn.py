"""
Tiny CNN library built directly on NumPy.

Why this file exists
--------------------
The project has no PyTorch/TensorFlow available in this environment, so we
implement just enough of a CNN stack to train the Cleaner and Router models:

    - Conv2D with zero padding
    - ReLU
    - Global average pooling
    - Masked average pooling (weighted pooling by a channel mask)
    - Fully-connected (Linear)
    - Softmax + cross-entropy loss
    - Mean squared error loss
    - Adam optimizer

Shapes follow PyTorch NCHW convention: (batch, channels, height, width).

All layers expose forward(x) and backward(grad) and collect trainable params
into Module.parameters() so the Adam optimizer can update them uniformly.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# im2col utilities
# ---------------------------------------------------------------------------

def _pad_input(x: np.ndarray, pad: int) -> np.ndarray:
    if pad == 0:
        return x
    return np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode="constant")


def im2col(x: np.ndarray, kh: int, kw: int, pad: int = 0) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """Return a (C*kh*kw, N*OH*OW) matrix suitable for matmul-based conv."""
    n, c, h, w = x.shape
    oh = h + 2 * pad - kh + 1
    ow = w + 2 * pad - kw + 1
    xp = _pad_input(x, pad)

    # Strided view of patches
    s_n, s_c, s_h, s_w = xp.strides
    shape = (n, c, kh, kw, oh, ow)
    strides = (s_n, s_c, s_h, s_w, s_h, s_w)
    patches = np.lib.stride_tricks.as_strided(xp, shape=shape, strides=strides)
    # (n, c, kh, kw, oh, ow) -> (c*kh*kw, n*oh*ow)
    cols = patches.transpose(1, 2, 3, 0, 4, 5).reshape(c * kh * kw, n * oh * ow)
    return cols, (n, oh, ow, c)


def col2im(cols: np.ndarray, x_shape: Tuple[int, int, int, int], kh: int, kw: int, pad: int = 0) -> np.ndarray:
    """Inverse of im2col. Accumulates gradient contributions into the padded input."""
    n, c, h, w = x_shape
    oh = h + 2 * pad - kh + 1
    ow = w + 2 * pad - kw + 1
    # (c*kh*kw, n*oh*ow) -> (c, kh, kw, n, oh, ow) -> (n, c, kh, kw, oh, ow)
    patches = cols.reshape(c, kh, kw, n, oh, ow).transpose(3, 0, 1, 2, 4, 5)

    xp = np.zeros((n, c, h + 2 * pad, w + 2 * pad), dtype=cols.dtype)
    for i in range(kh):
        for j in range(kw):
            xp[:, :, i:i + oh, j:j + ow] += patches[:, :, i, j, :, :]

    if pad == 0:
        return xp
    return xp[:, :, pad:pad + h, pad:pad + w]


# ---------------------------------------------------------------------------
# Base classes
# ---------------------------------------------------------------------------

class Parameter:
    __slots__ = ("data", "grad")

    def __init__(self, data: np.ndarray):
        self.data = data.astype(np.float32)
        self.grad = np.zeros_like(self.data)


class Module:
    def __init__(self):
        self._params: Dict[str, Parameter] = {}
        self._children: Dict[str, "Module"] = {}
        self._training = True

    def train(self, flag: bool = True):
        self._training = flag
        for child in self._children.values():
            child.train(flag)

    def eval(self):
        self.train(False)

    def register_param(self, name: str, param: Parameter):
        self._params[name] = param

    def register_module(self, name: str, module: "Module"):
        self._children[name] = module

    def parameters(self) -> List[Parameter]:
        out = list(self._params.values())
        for child in self._children.values():
            out.extend(child.parameters())
        return out

    def state_dict(self) -> Dict[str, np.ndarray]:
        out: Dict[str, np.ndarray] = {}
        for name, p in self._params.items():
            out[name] = p.data.copy()
        for cname, child in self._children.items():
            for k, v in child.state_dict().items():
                out[f"{cname}.{k}"] = v
        return out

    def load_state_dict(self, state: Dict[str, np.ndarray]):
        for name, p in self._params.items():
            if name in state:
                p.data = state[name].astype(np.float32).copy()
        for cname, child in self._children.items():
            subset = {k[len(cname) + 1:]: v for k, v in state.items() if k.startswith(cname + ".")}
            if subset:
                child.load_state_dict(subset)

    def zero_grad(self):
        for p in self.parameters():
            p.grad.fill(0.0)

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def backward(self, *args, **kwargs):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


# ---------------------------------------------------------------------------
# Layers
# ---------------------------------------------------------------------------

class Conv2D(Module):
    """2D convolution with zero padding."""

    def __init__(self, in_c: int, out_c: int, k: int = 3, pad: int = 1):
        super().__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.k = k
        self.pad = pad
        # He init
        fan_in = in_c * k * k
        std = math.sqrt(2.0 / fan_in)
        self.register_param("W", Parameter(np.random.randn(out_c, in_c, k, k).astype(np.float32) * std))
        self.register_param("b", Parameter(np.zeros((out_c,), dtype=np.float32)))
        self._cache: Optional[Tuple] = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        cols, (n, oh, ow, _) = im2col(x, self.k, self.k, pad=self.pad)
        W_row = self._params["W"].data.reshape(self.out_c, -1)
        out = W_row @ cols + self._params["b"].data[:, None]
        out = out.reshape(self.out_c, n, oh, ow).transpose(1, 0, 2, 3)
        self._cache = (x.shape, cols, W_row, n, oh, ow)
        return out

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        x_shape, cols, W_row, n, oh, ow = self._cache
        # (N, OC, OH, OW) -> (OC, N*OH*OW)
        g = grad_out.transpose(1, 0, 2, 3).reshape(self.out_c, -1)

        self._params["W"].grad += (g @ cols.T).reshape(self._params["W"].data.shape)
        self._params["b"].grad += g.sum(axis=1)

        grad_cols = W_row.T @ g
        grad_x = col2im(grad_cols, x_shape, self.k, self.k, pad=self.pad)
        return grad_x


class ReLU(Module):
    def __init__(self):
        super().__init__()
        self._mask: Optional[np.ndarray] = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._mask = (x > 0)
        return x * self._mask

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        return grad_out * self._mask


class Linear(Module):
    def __init__(self, in_f: int, out_f: int):
        super().__init__()
        std = math.sqrt(2.0 / in_f)
        self.register_param("W", Parameter(np.random.randn(in_f, out_f).astype(np.float32) * std))
        self.register_param("b", Parameter(np.zeros((out_f,), dtype=np.float32)))
        self._cache: Optional[np.ndarray] = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._cache = x
        return x @ self._params["W"].data + self._params["b"].data

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        x = self._cache
        self._params["W"].grad += x.T @ grad_out
        self._params["b"].grad += grad_out.sum(axis=0)
        return grad_out @ self._params["W"].data.T


class GlobalAvgPool2D(Module):
    def __init__(self):
        super().__init__()
        self._in_shape: Optional[Tuple] = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._in_shape = x.shape
        return x.mean(axis=(2, 3))

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        n, c, h, w = self._in_shape
        denom = float(h * w)
        grad = np.broadcast_to(
            grad_out[:, :, None, None] / denom, (n, c, h, w)
        ).copy()
        return grad


class MaskedAvgPool2D(Module):
    """Pooling weighted by a spatial mask that is part of the input tensor.

    Used so the network gets a view that is *specific* to some cells of interest
    (e.g. the candidate net's pins, the router head, the router sink). This keeps
    the network location-aware even after a conv stack.
    """

    def __init__(self):
        super().__init__()
        self._cache = None

    def forward(self, feat: np.ndarray, mask: np.ndarray) -> np.ndarray:
        # mask: (N, 1, H, W) in [0, 1]
        weight = mask
        denom = weight.sum(axis=(2, 3), keepdims=True).clip(min=1.0)
        pooled = (feat * weight).sum(axis=(2, 3), keepdims=True) / denom  # (N, C, 1, 1)
        self._cache = (feat, weight, denom)
        return pooled.squeeze(-1).squeeze(-1)

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        feat, weight, denom = self._cache
        # grad wrt feat only (mask is an input slice, we don't update it)
        n, c, h, w = feat.shape
        g = grad_out[:, :, None, None] * (weight / denom)
        return g


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def log_softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - x.max(axis=axis, keepdims=True)
    return x - np.log(np.exp(x).sum(axis=axis, keepdims=True))


def cross_entropy_with_logits(logits: np.ndarray, target: np.ndarray) -> Tuple[float, np.ndarray]:
    """target: int labels (N,). Returns (loss, grad wrt logits)."""
    n = logits.shape[0]
    lp = log_softmax(logits, axis=-1)
    loss = -lp[np.arange(n), target].mean()
    p = softmax(logits, axis=-1)
    grad = p.copy()
    grad[np.arange(n), target] -= 1.0
    grad /= n
    return float(loss), grad


def soft_cross_entropy(logits: np.ndarray, soft_target: np.ndarray, mask: Optional[np.ndarray] = None) -> Tuple[float, np.ndarray]:
    """soft_target: float (N, K) summing to 1 per row.  mask: optional bool (N, K)."""
    if mask is not None:
        logits = np.where(mask, logits, -1e9)
    lp = log_softmax(logits, axis=-1)
    loss = -(soft_target * lp).sum(axis=-1).mean()
    p = softmax(logits, axis=-1)
    grad = (p - soft_target) / soft_target.shape[0]
    if mask is not None:
        grad = np.where(mask, grad, 0.0)
    return float(loss), grad


def mse_loss(pred: np.ndarray, target: np.ndarray) -> Tuple[float, np.ndarray]:
    diff = pred - target
    loss = float(np.mean(diff * diff))
    grad = 2.0 * diff / diff.size
    return loss, grad


# ---------------------------------------------------------------------------
# Optimizer
# ---------------------------------------------------------------------------

class Adam:
    def __init__(self, params: List[Parameter], lr: float = 1e-3, betas=(0.9, 0.999), eps: float = 1e-8, weight_decay: float = 0.0):
        self.params = list(params)
        self.lr = lr
        self.b1, self.b2 = betas
        self.eps = eps
        self.wd = weight_decay
        self._m = [np.zeros_like(p.data) for p in self.params]
        self._v = [np.zeros_like(p.data) for p in self.params]
        self._t = 0

    def step(self):
        self._t += 1
        b1, b2 = self.b1, self.b2
        for i, p in enumerate(self.params):
            g = p.grad
            if self.wd > 0.0:
                g = g + self.wd * p.data
            self._m[i] = b1 * self._m[i] + (1 - b1) * g
            self._v[i] = b2 * self._v[i] + (1 - b2) * (g * g)
            m_hat = self._m[i] / (1 - b1 ** self._t)
            v_hat = self._v[i] / (1 - b2 ** self._t)
            p.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

    def zero_grad(self):
        for p in self.params:
            p.grad.fill(0.0)

    def set_lr(self, lr: float):
        self.lr = lr
