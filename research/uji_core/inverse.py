"""Transfer a local displacement of the historical inverse map to target anchors.

The model reuses raw 3-WKNN conventions (100 -> -110, RMS distance, inverse
distance vote). It can be deployed with only H, S, a query and frozen scalars.
"""
import numpy as np


def signal(rss):
    # +110 shift makes undetected entries exactly zero and preserves distances.
    return np.where(rss == 100, 0, rss.astype(np.float64) + 110.)


def squared(a, b):
    return np.maximum((a*a).sum(1)[:, None] + (b*b).sum(1) - 2*(a @ b.T), 0.)


def order(distance, ids, n):
    # Equal distances always resolve by original reference CSV row, even H vs S.
    return np.lexsort((np.broadcast_to(ids, distance.shape), distance), axis=1)[:, :n]


def vote(distances, xy, width):
    weights = 1. / np.maximum(np.sqrt(distances / max(width, 1)), 1e-8)
    weights /= weights.sum(1, keepdims=True)
    return np.einsum("qr,qrd->qd", weights, xy), weights


class HistoryMap:
    def __init__(self, floors):
        self.signal = signal(np.concatenate([h["rss"] for h in floors]))
        self.visible = self.signal.any(0)
        self.xy = np.concatenate([h["xy"] for h in floors])
        self.ids = np.concatenate([h["ids"] for h in floors])
        self._visible_signal = self.signal[:, self.visible].copy()

    def match(self, features, n):
        """Cache neighbors in H only. Future S and query labels cannot change them."""
        d = squared(features[:, self.visible], self._visible_signal)
        ix = order(d, self.ids, n)
        return np.take_along_axis(d, ix, axis=1), ix


class Episode:
    def __init__(self, bank, query, support, sxy, sids, qmatch, smatch):
        self.sxy = sxy
        hv = bank.visible
        sv = support.any(0)
        visible = hv | sv
        self.width = int(visible.sum())
        hd, indices = qmatch
        self.hxy = bank.xy[indices]
        self.hids = bank.ids[indices]
        self.sids = sids
        # APs seen only in current S add the same squared offset to each H row.
        self.hd = hd + (query[:, sv & ~hv]**2).sum(1)[:, None]
        self.sd = squared(query[:, visible], support[:, visible])
        self.qmatch = hd, self.hxy
        self.smatch = smatch[0], bank.xy[smatch[1]]
        self.hwidth = int(hv.sum())

    def anchors(self, k):
        return tuple(vote(d[:, :k], xy[:, :k], self.hwidth)[0] for d, xy in (self.qmatch, self.smatch))

    def local(self, k):
        ix = order(self.sd, self.sids, k)
        return vote(np.take_along_axis(self.sd, ix, axis=1), self.sxy[ix], self.width)[0]

    def predict_parts(self, k, scale, anchors):
        """F(q) = direct weighted vote + gamma * supported history displacement."""
        hx = self.hxy[:, :k]
        nr = hx.shape[1]
        d = np.column_stack((self.hd[:, :k], scale**2 * self.sd))
        ids = np.column_stack((self.hids[:, :k], np.broadcast_to(self.sids, self.sd.shape)))
        ix = order(d, ids, k)
        all_xy = np.concatenate((hx, np.broadcast_to(self.sxy, (len(hx), *self.sxy.shape))), axis=1)
        direct, weights = vote(np.take_along_axis(d, ix, axis=1),
                               np.take_along_axis(all_xy, ix[:, :, None], axis=1), self.width)
        hq, hs = anchors
        support_ix = np.maximum(ix - nr, 0)
        shift = hq[:, None] - hs[support_ix]
        support_weights = weights * (ix >= nr)
        displacement = np.einsum("qr,qrd->qd", support_weights, shift)
        return direct, displacement, support_weights.sum(1)


def predict(history_floors, support, query_rss, *, neighbors, scale, transport):
    """Public inference entry. H/S IDs identify training rows and resolve ties."""
    bank = HistoryMap(history_floors)
    q, s = signal(query_rss), signal(support["rss"])
    qm = bank.match(q, neighbors)
    if len(s) == 0:
        return vote(qm[0], bank.xy[qm[1]], bank.visible.sum())[0]
    episode = Episode(bank, q, s, support["xy"], support["ids"], qm, bank.match(s, neighbors))
    direct, shift, _ = episode.predict_parts(neighbors, scale, episode.anchors(neighbors))
    return direct + transport * shift
