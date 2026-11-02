# Version and data ledger: no existing version replaced

This study is additive on `research/uji-core-refactor-20261101`. `main`, `refi.py`, the paper replay, its original source weights and previous experiment results were not edited. `archive/refi-before-local-shape-20261102` names the state immediately before this study. It is not a mirror of the advancing research branch. Git records searchable history; the points below are convenient entry points, not a relabeling of past results.

| Restore point | Contents |
|---|---|
| `0ddfcad816a125c2269960dc814bc32a7d4bb998` | main head observed during this study |
| `156efc6622793f88693b2d5e58cd442e9e512707` | first all-floor nested UJI refactor |
| `49f581bd11f1500b9ae4e9787af98f3ff445ef55` | round 2, closing anchor interpolation |
| `50de2270e16231a057771447cf8f20f4eb8abcf1` | Minho single-scan ordering contrast |
| `683a2974756cd3678bca7da43511f7ddb4195840` | v0.26 original-protocol replay |
| `e795deae308671c173f135aa07467523a93a078a` | variance and sparse computation ablation |
| `b7f7dbcdee8e577287504a85b005421e97e35278` | original GP / seven-coefficient / nonlinear comparison; pretrained weights and prediction archives; pre-study snapshot |
| `28dc91c2a1cf7d4ace9bec9e177e00fd976afbd3` | See corrected full hash from the `PROTOCOL.md` creation commit using `git log --diff-filter=A -- research/local_shape_v1/PROTOCOL.md`; short ID `28dc91c` |
| `73ab8cb91fca3f195046d0718da637802be56aac` | frozen executable study submitted to Actions |

**Offline restore.** [Backup run](https://github.com/Amanon-666/law/actions/runs/41406681770) created a self-contained bundle of the 78 commits reachable from the pre-study snapshot. Clean clone and `git fsck --full` were checked, as were the six milestone ancestors and four previous weight/prediction ZIPs. No claim is made about untracked files on personal machines or unrelated branches. The artifact contains `COMMITS.txt`, `MILESTONES.txt`, `CHECKPOINTS.txt`, and `SHA256SUMS`. From the unpacked folder:

```sh
sha256sum -c SHA256SUMS
git clone -b restore/refi-pre-shape ./refi-pre-shape.bundle restored-law
git -C restored-law rev-parse HEAD
```

**Sources.** UTSIndoorLoc: Song et al., *A Novel Convolutional Neural Network Based Indoor Localization Framework With WiFi Fingerprinting*, dataset authors' [file repository](https://github.com/XudongSong/UTSIndoorLoc-dataset); used only `UTS_training.csv`. Tampere 2017: Lohan, Torres-Sospedra, Gonzalez, [original record](https://doi.org/10.5281/zenodo.5174851); read only the two buildings' training RSS and coordinate file members, not the official test records. Hashes of both source downloads are pinned in `experiment.py`. The design is a constrained cross-building diagnostic, not a claim of reproducing the historical full-AP end-to-end UJI scores or an external blind confirmation.

**Rerun.** Environment is pinned in `.github/workflows/refi-local-shape-v1.yml`. With the source files named `uts.csv` and `tampere.zip`, from repository root run the following. The archive of the original seven coefficients is loaded and hash checked; the neural stage never backpropagates through the target GP.

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python -m research.local_shape_v1.experiment UTS uts.csv out/fit
python -m research.local_shape_v1.experiment TIE1 tampere.zip out/TIE1 --weights out/fit
python -m research.local_shape_v1.experiment SAH1 tampere.zip out/SAH1 --weights out/fit
```

[Primary workflow](https://github.com/Amanon-666/law/actions/runs/41406596230). To recompute errors without fitting, unzip the `replication.zip` output and run `python research/local_shape_v1/verify.py PATH_TO_EXTRACTED_FILES` with NumPy. Reported times are Python process timings (not CI queue, software installation, network transfer, or estimates for a particular GPU).