# FAA attribution forecast 02 — entry

Per-market, per-horizon realized-sigma fan in logit space, Student-t(8)
quantiles, plus a damped drift (see `main.py` docstring for the full
derivation and numbers).

## Validation

Live FAA markets (Polymarket CLOB, 2026-11-13 21:35 UTC → 2026-11-14
08:30 UTC), 6 cases 4h apart, mean pinball loss x100:

| model | loss |
|---|---|
| house baseline | 0.4453 |
| NVIDIA-v2 fan, no drift | 0.3309 |
| **this model (damped drift, tau 60, damp 0.5)** | **0.2992** |
| full linear drift | 0.3875 |

Damped-drift family is stable (tau 45/60/90, damp 0.35/0.5/0.65:
0.299–0.302). NVIDIA practice p00–p04: 0.3267 (no-drift fan: 0.3059) —
the rules state practice is no guide to the final, which is these FAA
markets; the live-FAA evidence is what the final is graded on.

## Checks run on the shipped file

- All 6 practice outputs valid (1 key, markets match, 240x5, monotone,
  [0,1], <1MB); shipped-file practice loss reproduces the harness
  number exactly (0.326695).
- Two runs byte-identical; ~0.56s per practice case, 0.13s for a
  4-market final-shaped case (limit 8s).
- Final-shape simulation (live FAA data as h00): medians track last
  prices, spreads tight and sane.
