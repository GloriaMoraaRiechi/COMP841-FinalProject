"""
CNN models for the Cleaner and Router.

Cleaner scoring CNN:
    Input:  (N, 12, H, W)  feature tensor for ONE candidate net's before/after
    Output: (N,)           a single scalar score per candidate, higher is better

Router policy-value CNN:
    Input:  (N, 9, H, W)   feature tensor for the router's state
    Output: (logits, value) where logits is (N, 4) for the 4 actions (UP/DOWN/LEFT/RIGHT)
            and value is (N,) a scalar estimate of return-to-go

Both networks are fully convolutional so they work on any HxW grid.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from nn import Conv2D, ReLU, Linear, Module, GlobalAvgPool2D, MaskedAvgPool2D


class ConvBlock(Module):
    def __init__(self, in_c: int, out_c: int):
        super().__init__()
        self.conv = Conv2D(in_c, out_c, k=3, pad=1)
        self.relu = ReLU()
        self.register_module("conv", self.conv)
        self.register_module("relu", self.relu)

    def forward(self, x: np.ndarray) -> np.ndarray:
        return self.relu(self.conv(x))

    def backward(self, g: np.ndarray) -> np.ndarray:
        g = self.relu.backward(g)
        g = self.conv.backward(g)
        return g


class ConvBackbone(Module):
    def __init__(self, in_channels: int, width: int, depth: int):
        super().__init__()
        self.blocks: List[ConvBlock] = []
        self.blocks.append(ConvBlock(in_channels, width))
        self.register_module("block0", self.blocks[-1])
        for d in range(depth - 1):
            self.blocks.append(ConvBlock(width, width))
            self.register_module(f"block{d+1}", self.blocks[-1])

    def forward(self, x: np.ndarray) -> np.ndarray:
        h = x
        for b in self.blocks:
            h = b(h)
        return h

    def backward(self, g: np.ndarray) -> np.ndarray:
        for b in reversed(self.blocks):
            g = b.backward(g)
        return g


# ---------------------------------------------------------------------------
# Cleaner
# ---------------------------------------------------------------------------

class CleanerScoringCNN(Module):
    """Scores one candidate net at a time. Higher = more likely the right net
    to rip up and reroute next."""

    def __init__(self, in_channels: int = 12, width: int = 32, depth: int = 3):
        super().__init__()
        self.width = width
        self.backbone = ConvBackbone(in_channels, width=width, depth=depth)
        self.gap = GlobalAvgPool2D()
        self.pool_cand_before = MaskedAvgPool2D()
        self.pool_pins = MaskedAvgPool2D()
        self.pool_overlap = MaskedAvgPool2D()
        self.pool_cand_after = MaskedAvgPool2D()
        self.fc1 = Linear(width * 5, width * 2)
        self.relu1 = ReLU()
        self.fc2 = Linear(width * 2, width)
        self.relu2 = ReLU()
        self.fc3 = Linear(width, 1)

        for name in ("backbone", "gap", "pool_cand_before", "pool_pins",
                     "pool_overlap", "pool_cand_after",
                     "fc1", "relu1", "fc2", "relu2", "fc3"):
            self.register_module(name, getattr(self, name))

        # cached for backward
        self._cache = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        feat = self.backbone(x)
        g = self.gap(feat)                                          # (N, C)
        pb = self.pool_cand_before(feat, x[:, 0:1])                 # (N, C)
        pp = self.pool_pins(feat, x[:, 2:3])                        # (N, C)
        po = self.pool_overlap(feat, x[:, 6:7])                     # (N, C)
        pa = self.pool_cand_after(feat, x[:, 7:8])                  # (N, C)

        fused = np.concatenate([g, pb, pp, po, pa], axis=1)         # (N, 5C)
        h = self.relu1(self.fc1(fused))
        h = self.relu2(self.fc2(h))
        out = self.fc3(h).squeeze(-1)
        self._cache = (feat.shape,)
        return out

    def backward(self, grad: np.ndarray):
        # grad: (N,) -> (N, 1)
        grad = grad[:, None]
        grad = self.fc3.backward(grad)
        grad = self.relu2.backward(grad)
        grad = self.fc2.backward(grad)
        grad = self.relu1.backward(grad)
        grad = self.fc1.backward(grad)                              # (N, 5C)
        c = self.width
        g_g = grad[:, 0:c]
        g_pb = grad[:, c:2*c]
        g_pp = grad[:, 2*c:3*c]
        g_po = grad[:, 3*c:4*c]
        g_pa = grad[:, 4*c:5*c]

        # Sum gradients flowing back to feat
        gg = self.gap.backward(g_g)
        gpb = self.pool_cand_before.backward(g_pb)
        gpp = self.pool_pins.backward(g_pp)
        gpo = self.pool_overlap.backward(g_po)
        gpa = self.pool_cand_after.backward(g_pa)
        g_feat = gg + gpb + gpp + gpo + gpa

        self.backbone.backward(g_feat)


# ---------------------------------------------------------------------------
# Router policy-value network
# ---------------------------------------------------------------------------

class RouterPolicyValueNet(Module):
    """CNN that consumes a router state and outputs (4 action logits, scalar value)."""

    def __init__(self, in_channels: int = 9, width: int = 32, depth: int = 3, num_actions: int = 4):
        super().__init__()
        self.width = width
        self.num_actions = num_actions
        self.backbone = ConvBackbone(in_channels, width=width, depth=depth)
        self.gap = GlobalAvgPool2D()
        self.pool_head = MaskedAvgPool2D()
        self.pool_sink = MaskedAvgPool2D()

        fused_dim = width * 3
        self.policy_fc1 = Linear(fused_dim, width)
        self.policy_relu = ReLU()
        self.policy_fc2 = Linear(width, num_actions)
        self.value_fc1 = Linear(fused_dim, width)
        self.value_relu = ReLU()
        self.value_fc2 = Linear(width, 1)

        for name in ("backbone", "gap", "pool_head", "pool_sink",
                     "policy_fc1", "policy_relu", "policy_fc2",
                     "value_fc1", "value_relu", "value_fc2"):
            self.register_module(name, getattr(self, name))

        self._cache = None

    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        feat = self.backbone(x)
        g = self.gap(feat)
        ph = self.pool_head(feat, x[:, 3:4])
        ps = self.pool_sink(feat, x[:, 4:5])
        fused = np.concatenate([g, ph, ps], axis=1)

        # Keep the fused shared input; cache it so we can send gradient back from both heads.
        p = self.policy_relu(self.policy_fc1(fused))
        logits = self.policy_fc2(p)
        v = self.value_relu(self.value_fc1(fused))
        value = self.value_fc2(v).squeeze(-1)

        self._cache = (feat.shape,)
        return logits, value

    def backward(self, grad_logits: np.ndarray, grad_value: np.ndarray):
        # Policy head backward
        g1 = self.policy_fc2.backward(grad_logits)
        g1 = self.policy_relu.backward(g1)
        g1 = self.policy_fc1.backward(g1)

        # Value head backward
        gv = self.value_fc2.backward(grad_value[:, None])
        gv = self.value_relu.backward(gv)
        gv = self.value_fc1.backward(gv)

        g_fused = g1 + gv
        c = self.width
        g_g = g_fused[:, 0:c]
        g_ph = g_fused[:, c:2*c]
        g_ps = g_fused[:, 2*c:3*c]

        gg = self.gap.backward(g_g)
        gph = self.pool_head.backward(g_ph)
        gps = self.pool_sink.backward(g_ps)
        g_feat = gg + gph + gps
        self.backbone.backward(g_feat)
