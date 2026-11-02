# R1 original single-scan, three-history benchmark — fixed bounded shard

Scope fixed before successful prediction scores. Read the R1 evidence ZIP as the sole manifest of 0-based global CSV data rows; do not generate new S/Q. Code: score.py. Initial reproducibility/row-shape audit: workflow refi-r1-fixed-evidence.yml. Evaluation: refi-r1-bounded.yml.

## Fixed experiment

- B1 target floors 0,1,2,3. On every target floor use **all other three B1 history floors**, not the two-source F0/F3 setup of the manuscript Table II. Horizontal coordinates in UJI original U; no meters conversion in these results.
- Original draw **0 only** per floor, two disjoint evaluation regimes (official public validation and strict whole-handset-held-out internal). Support is the identical 20-position ordered list in evidence.zip for that target/regime, nested prefixes K=1,3,5. Exactly **one recorded RSS acquisition per position**. All 4 x 2 x 3 = 24 event keys must be reported or listed missing. This is **not** the 400-key full R1 (10 draws, 5 budgets, 2 regimes), and is not a complete monotonicity/generalization verdict.
- Official query exact original validationData.csv row IDs, expected B1 F0/F1/F2/F3 rows 30/143/87/47. Strict-phone query exactly 12 fixed training-file coordinates per floor; selected held-out handset 14/7/23/10, excluded in all historical floors and target Support. No training or target fitting on Q.
- Historical AP set in each event: union of those detected in the three permitted H floors and current K-support. No query AP may affect selection. At each K use visible H/S to define the same 2 U coordinate-grid rule across GP, seven, and nonlinear. The grid may expand with K as H/S expands; report its size.
- Each historical AP field fitted exactly once per floor/regime with original frozen field hyperparameters and non-Gaussian detection model, 640 inference iterations per source/AP; cache across the three K prefixes. APs missing all three sources use the existing local fallback.
- Shared three-source simplex weights fitted **only on the current S** by multi_source_adaptation.fit_source_weights from the raw uncorrected GP; one shared vector for all APs. Fix it while applying all three arms: plain GP; seven source-trained coefficients (SHA256 cd5c4972...); original source-trained nonlinear OperatorNet V1 mean only (SHA256 53061b01...). No learned variance head. The learned rules were fitted on B0/B2 source episodes with two historical layers; deploying them unchanged with J=3 is a real zero-shot source-count generalization test.
- Re-run the same non-Gaussian support posterior per arm, with reference M-PMC for K<=5 and ESS >=1000. Weight fitting on S and support-field conditioning on S are one plug-in EB pipeline, not independent likelihood factors. Predict exact RSS/detection/missingness on a common candidate grid and use posterior coordinate mean.
- All other numeric choices are frozen to existing repository code, no B1 hyperparameter tuning, post-hoc algorithm selection, changing missing-value treatment, or new spatial kernel. No unseen external confirmation sets are accessed.

## Controls and decision

- Do **not** rerun the previously archived R1 WKNN experiments: evidence.zip already contains all 650 original 5-way WKNN budget/draw combinations and all 650 device-stress groups. Extract the source-selected 7-neighbor prototype WKNN, raw 15-neighbor, and the fixed three-neighbor reference from wknn_events.csv; phone retained comparators strong_wknn, fixed_tobit, tuned_tobit, empirical from device_stress_events.csv.
- The archived source-selected 7-neighbor prototype method should **not** be called the only strong WKNN: on previously seen B1 official validation, the fixed-3 prototype is better across all five K macro means. Both must appear as controls, and no choice may be made based on that B1 ranking.
- K<=5 results are a bounded diagnostic. No gate is passed without matched 10-draw K1/3/5/10/20, corresponding official and strict phone, consistent K5 backend bridge for K20, same-map WKNN (observable filled conditional RSS, not latent f), independent held-out buildings, and pre-registered external confirmation.
- For each event record query IDs/hash, history floor IDs, AP count, Support IDs/hash, S-only source weights, actual MDE and P90 (position then event/floor balanced), min ESS, compute seconds, and any missing/failed runs. Failures count against completeness. Never report only completed favorable events. Do not compare this R1 K1/3/5 R1 curve numerically to the manuscript K5R3 two-history table as if identical.
- A new model is not promoted unless **one frozen version** separately meets the user's three final criteria and manuscript Table-II dual gates (<=10.128/11.007 U), and independent numerical reproduction plus genuine unseen-site confirmation succeeds.

## Prior evidence check

- Original trainingData.csv SHA256 45ca0128bd12019c976bb4793e407c979c82995d7a5940ab7288620247905168.
- Original R1 manifest member SHA256 5f4fb0ceccc5a64aa7411cc287715e6fa9a39d593da6713910920a3fda67c613.
- Original strict-phone design SHA256 76763a87569d885705dccc54cbd72e878f7a711cbb5343fbfee4517caf2482e6.
- Initial run failed before history fitting because the nonlinear archive stored operator.pt at the ZIP root, not under source/. This implementation path was fixed without touching frozen bytes or changing any scoring choices. Preserve that failure in the run history.
