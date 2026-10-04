"""Reproduces the Month 6 Quant Lab result: what drawdown-triggered selling costs."""
import numpy as np

from analytical_investor import behavior, simulate

paths = simulate.student_t_returns(480, 0.07 / 12, 0.16 / np.sqrt(12), size=2000, seed=7)
out = behavior.behavior_gap(paths, years=40)
for k, v in out.items():
    print(f"{k:>15}: {v:.3f}")
