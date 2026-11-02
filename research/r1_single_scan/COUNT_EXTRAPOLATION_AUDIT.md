# R1 frozen nonlinear checkpoint: source-count extrapolation audit (2026-11-02)

## Verified structural constraint — no target-Q selection

The archived V1 training code `research/uji_core/round2/paper_replay/limit_study/source.py` uses `operator_prior_v1/run_operator_prior.py::build_tasks`. The latter takes **all pairs** of history floors for each source task; every one of 42 B0/B2 training tasks feeds **J=2** history tokens to the same `OperatorNet`. Its forward pass (`operator_prior_v1/operator_prior.py`) includes a separate scalar `log1p(tokens.shape[0])` concatenated to pooled features. The original R1 B1 target uses **J=3** histories. The frozen R1 scoring code does not retrain the model and therefore supplies `log(4)` where the training data supplied only `log(3)`.

For the first hidden layer the count contribution is `w_c log(1+J)+b`. On all training examples, `w_c` and `b` are unidentifiable separately: `(w_c+t, b-t log(3))` yields identical training predictions for any `t`, but changes J=3 predictions by `t log(4/3)`. Thus no source training evidence identifies the correct three-source count response. This is a **real extrapolation confound**, not evidence that all nonlinear prior-mean rules fail.

The original immutable source-V1 checkpoint was downloaded read-only from successful GitHub run `41372212532`, artifact `source-v1` ID `13111055882`; its SHA256 in the earlier frozen protocol is `53061b01164f78def406e7b1395c5e62a295adb9a9a3570a2b91efed7b71e7c7`. Direct tensor inspection: `rho.0.weight[:, -1]` has L2 norm **1.102820**, max absolute coefficient **0.333342**. The count change `log(4)-log(3)=0.287682` adds a vector with L2 norm **0.317262** to the 32 first-hidden-layer preactivations, holding pooled source features fixed. This does **not** quantify the resulting real-world RSS shift, nor prove that the source-count input caused the localization regressions.

## Locked R1 implications

The existing 8-shard/24-cell draw-0 run `41443707993` must remain unchanged. Report the frozen V1 as the **actual J=2-trained to J=3 deployment stress**, not as a clean architecture comparison against seven coefficients. Historical GP, seven, and nonlinear have identical H/S/AP/Q/grid and posterior paths. In the initial four successful shards (official F0/F1; held-out-phone F0/F1), independently re-checking the archived query rows showed e.g. official F0 K5: GP **15.098**, seven **11.265**, frozen nonlinear **17.698 U**; held-out-phone F0 K5: GP **15.329**, seven **14.662**, nonlinear **12.522 U**. These two regimes have different Q and are **not paired across regimes**. No four-floor or K-monotonicity conclusion follows from these partial results.

The matching read-only validator `research/r1_single_scan/verify_locked.py` confirms each available shard's exact query-id hash, immutable source checkpoint hashes, nested support prefixes, three actual history floor IDs, one real scan per support position, candidate predictions, position-weighted MDE and P90, and ESS. It refuses to report a completed 24-cell result while any shard is missing.

A subsequent *source-only* design, if justified after the bounded R1 is closed, must deliberately train/source-validate variable J=2/3 tasks or remove the unidentifiable count input **without** choosing a correction from B1 test results. Do not retrospectively alter the present checkpoint, its locked predictions, or the strong reference comparisons. Do not open UJIUB or Minho holdouts.

[Pre-registered R1 protocol](PROTOCOL.md) · [Frozen source reconstruction](../uji_core/round2/paper_replay/limit_study/source.py) · [OperatorNet implementation](../../operator_prior_v1/operator_prior.py) · [R1 fixed run](https://github.com/Amanon-666/law/actions/runs/41443707993).
