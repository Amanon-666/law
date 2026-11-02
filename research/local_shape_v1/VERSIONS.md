# Version and data ledger

This study continues `research/uji-core-refactor-20261101` by adding a separate research directory. Neither `main`, `refi.py`, nor original studies, results or checkpoints were edited. `archive/refi-before-local-shape-20261102` names the state immediately before this study; do not advance that ref.

| Restorable commit | What it contains |
|---|---|
| `0ddfcad816a125c2269960dc814bc32a7d4bb998` | Main as observed during this study |
| `156efc6622793f88693b2d5e58cd442e9e512707` | First all-floor nested UJI refactor |
| `49f581bd11f1500b9ae4e9787af98f3ff445ef55` | Round 2, closing anchor interpolation |
| `50de2270e16231a057771447cf8f20f4eb8abcf1` | Minho single-scan ordering contrast |
| `683a2974756cd3678bca7da43511f7ddb4195840` | v0.26 original-protocol replay |
| `e795deae308671c173f135aa07467523a93a078a` | Variance and sparse computation ablation |
| `b7f7dbcdee8e577287504a85b005421e97e35278` | GP / seven-coefficient / nonlinear comparison, trained weights, predictions; pre-study snapshot |
| `28dc91c2a1cf7d4ace9bec9e8a5325a9aeaf80d8` | Freeze local-shape protocol before scoring |
| `73ab8cb91fca3f195046d0718da637802be56aac` | Frozen executable study submitted to Actions |

**Offline restore.** [Backup run](https://github.com/Amanon-666/law/actions/runs/41406681770) produced a stand-alone bundle including all 78 commits reachable from the pre-study snapshot. A clean clone, `git fsck --full`, the six research milestone ancestors and four previous weight/prediction ZIPs were verified. This does not cover untracked files on another machine or unrelated branches. The archive includes `COMMITS.txt`, `MILESTONES.txt`, `CHECKPOINTS.txt`, `SHA256SUMS`. From its unpacked folder:

```sh
sha256sum -c SHA256SUMS
git clone -b restore/refi-pre-shape ./refi-pre-shape.bundle restored-law
git -C restored-law rev-parse HEAD
```

**Data.** Song et al., *A Novel Convolutional Neural Network Based Indoor Localization Framework With WiFi Fingerprinting*, [authors' dataset](https://github.com/XudongSong/UTSIndoorLoc-dataset): only `UTS_training.csv`. Lohan, Torres-Sospedra, Gonzalez, [Tampere 2017](https://doi.org/10.5281/zenodo.5174851): only the two buildings' training RSS/coordinate files; official test records not read. Hashes are pinned in `experiment.py`. This capped cross-building development test is not the full-AP manuscript protocol or independent external confirmation.

**Recompute the study.** Use `.github/workflows/refi-local-shape-v1.yml` for the environment. Original seven-coefficient checkpoint is read and verified; target GP inference is not traversed during learner training.

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python -m research.local_shape_v1.experiment UTS uts.csv out/fit
python -m research.local_shape_v1.experiment TIE1 tampere.zip out/TIE1 --weights out/fit
python -m research.local_shape_v1.experiment SAH1 tampere.zip out/SAH1 --weights out/fit
```

[Primary run](https://github.com/Amanon-666/law/actions/runs/41406596230). To check the numbers without a model fit, unzip `results/<run>-<attempt>/replication.zip` and run `python research/local_shape_v1/verify.py PATH_TO_EXTRACTED_FILES` using NumPy. Timings are measured Python process stages, not CI startup, software installation, network transfer or GPU time.