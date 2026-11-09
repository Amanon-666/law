"""Correlated multi-period histories -> one posterior at the latest history date.

The stationary separable temporal GP is a mature multitask construction, not
a new network. Raw scans keep their (period, row) identities. Only aliases of
the same acquisition are removed; identical-valued separate scans are kept.
"""
from datetime import datetime

import numpy as np
import torch

from data import group_scans
from calibration_relation import CalibrationRelation
from inference import _chol, infer
from run import ap_stats, tensor
from transfer import History, rbf


def source_calendar(path, base):
    # Dates beyond the established source validation period are not used.
    with np.load(path, allow_pickle=False) as arrays:
        dates = arrays["dates"][:12].tolist()
    parsed = [datetime.strptime(x, "%y%m%d") for x in dates]
    days = np.array([(x - parsed[0]).days for x in parsed], dtype=float)
    unit_days = float(np.median(np.diff(days[:11])))
    # Match the OLD source-estimated one-interval increment variance exactly:
    # Var[h(t+unit)-h(t)] = 2 A^2 (1-rho) = drift^2.
    rho = 1 - base["drift"] ** 2 / (2 * base["amplitude"] ** 2)
    if not 0 < rho < 1:
        raise ValueError("source increment variance cannot initialize stationary OU correlation")
    return dict(dates=dates, days=days.tolist(), unit_days=unit_days,
                temporal_length_days=float(-unit_days / np.log(rho)))


def collect_histories(floors, selections):
    """Selections are (period, original row IDs), not RSS values to deduplicate."""
    grouped = {}
    for period, row_ids in selections:
        grouped.setdefault(int(period), set()).update(map(int, row_ids))
    records = []
    for period, row_ids in sorted(grouped.items()):
        rows = np.array(sorted(row_ids), dtype=int)
        floor = floors[period]
        if not len(rows) or rows[0] < 0 or rows[-1] >= len(floor["rss"]):
            raise ValueError("history acquisition IDs outside their source period")
        scans = group_scans(floor["rss"][rows], floor["xy"][rows])
        records.append(dict(period=period, row_ids=rows, scans=scans))
    if not records:
        raise ValueError("multi-history study requires at least one historical acquisition")
    return records


def temporal_covariance(x, times_x, y, times_y, base, length_days):
    spatial = rbf(x, y, base["amplitude"], base["length"])
    return spatial * torch.exp(-(times_x[:, None] - times_y[None]).abs() / length_days)


def project_posterior(prior_cov, cross, posterior, target_cov, mean):
    """Integrate a joint Gaussian approximation; do not multiply expert posteriors."""
    projection = torch.cholesky_solve(cross.T, _chol(prior_cov)).T
    m = mean + projection @ (posterior["mean"] - mean)
    c = target_cov + projection @ (posterior["cov"] - prior_cov) @ projection.T
    return dict(mean=m, cov=(c + c.T) / 2)


class MultiHistory(History):
    def __init__(self, records, ap, base, obs, calendar, steps=160):
        periods = [r["period"] for r in records]
        if len(periods) != len(set(periods)):
            raise ValueError("merge duplicate source-period records with collect_histories first")
        positions = [tensor(r["scans"]["positions"]) for r in records]
        x = torch.cat(positions)
        times = torch.cat([p.new_full((len(p),), calendar["days"][r["period"] - 1])
                           for p, r in zip(positions, records)])
        counts = [ap_stats(r["scans"], ap) for r in records]
        stats = {key: torch.cat([c[key] for c in counts]) for key in counts[0]}
        cov = temporal_covariance(x, times, x, times, base, calendar["temporal_length_days"])
        posterior = infer(x.new_full((len(x),), base["mean"]), cov, stats, obs, steps=steps)
        z = torch.unique(x, dim=0)
        latest_time = times.max().item()
        target_time = z.new_full((len(z),), latest_time)
        cross = temporal_covariance(z, target_time, x, times, base, calendar["temporal_length_days"])
        target_cov = rbf(z, z, base["amplitude"], base["length"])
        projected = project_posterior(cov, cross, posterior, target_cov, base["mean"])
        super().__init__(z, projected, base)
        self.periods = periods
        self.latest_period = max(periods)
        self.scan_count = sum(len(r["row_ids"]) for r in records)
        self.temporal_length_days = calendar["temporal_length_days"]


def single_history(record, ap, base, obs, steps=160):
    """Original single-H rule, including its no-detection fallback."""
    scans = record["scans"]
    if not (scans["stats"]["k"][:, ap] > 0).any():
        return None
    z = tensor(scans["positions"])
    posterior = infer(z.new_full((len(z),), base["mean"]),
                      rbf(z, z, base["amplitude"], base["length"]),
                      ap_stats(scans, ap), obs, steps=steps)
    return History(z, posterior, base)


class GaussianStrengthRelation(CalibrationRelation):
    """Exact expected strength RBF under a Gaussian historical posterior.

    Same three parameters as the old kernel. Includes interpolation uncertainty
    and avoids a cubature rule depending on the number of historical positions.
    """
    def kernel(self, x, y, history, base):
        px, py = history.projection(x), history.projection(y)
        delta_cov = history.c - history.kernel(history.z, history.z)
        mx = history.base["mean"] + px @ (history.m - history.base["mean"])
        my = history.base["mean"] + py @ (history.m - history.base["mean"])
        cx = history.base["amplitude"] ** 2 + (px @ delta_cov * px).sum(-1)
        cy = history.base["amplitude"] ** 2 + (py @ delta_cov * py).sum(-1)
        cxy = history.kernel(x,y) + px @ delta_cov @ py.T
        v = cx[:,None] + cy[None] - 2*cxy
        if bool((v < -1e-7).any()):
            raise RuntimeError("historical Gaussian difference has negative variance")
        v = v.clamp_min(0)
        ell2 = self.log_strength_length.exp().square()
        relation = torch.sqrt(ell2/(ell2+v)) * torch.exp(-.5*(mx[:,None]-my[None]).square()/(ell2+v))
        local = self.logit_local.sigmoid()
        spatial = rbf(x,y,1.,self.log_length.exp())
        return base["drift"] ** 2 * (local*spatial+(1-local)*relation)
