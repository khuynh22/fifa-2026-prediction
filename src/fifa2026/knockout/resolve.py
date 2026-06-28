from __future__ import annotations
import numpy as np

def shootout_prob(pen_a: float, pen_b: float, depth_a: float = 0.0, depth_b: float = 0.0) -> float:
    """Near-even, nudged by penalty record (pen_*) and squad depth (depth_*)."""
    logit = 1.2 * (pen_a - pen_b) + 0.3 * (depth_a - depth_b)
    p = 1.0 / (1.0 + np.exp(-logit))
    return float(np.clip(p, 0.05, 0.95))

def resolve_tie(p_reg, pen_a: float = 0.5, pen_b: float = 0.5,
                depth_a: float = 0.0, depth_b: float = 0.0) -> float:
    p_reg = np.asarray(p_reg, dtype=float)
    p_win, p_draw, _p_loss = p_reg
    s = shootout_prob(pen_a, pen_b, depth_a, depth_b)
    return float(p_win + p_draw * s)
