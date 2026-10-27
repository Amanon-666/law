"""Linux resume adapter: relocate a frozen Mac checkpoint path, change no model.

Invoke only on ray-web. Existing full_ap_v1 manifests/checkpoints are reused.
The original freeze and code snapshots remain untouched.
"""
from pathlib import Path
import socket

import full_ap_study as study


if __name__ == "__main__":
    if socket.gethostname() != "ray-web":
        raise RuntimeError("This authorized resume entry point is server-only (ray-web)")
    original = study.load_mean
    def relocated_mean(setting):
        adapted = dict(setting)
        adapted["mean_checkpoint"] = str(Path(__file__).resolve().parent /
                                        "results/source_transfer_v4_spatial_control/affine.pt")
        return original(adapted)
    study.load_mean = relocated_mean
    study.main()
