"""Regression checks for the deployable path; no target data required."""
import numpy as np

from external_baselines import tpm_adapter as tpm
from refi import fit_map


def main():
    h = [{"xy": np.array([[0., 0.], [0., 0.], [30., 0.]]),
          "rss": np.array([[-57, 100, 100], [-59, 100, 100], [-89, -60, 100]])}]
    s = dict(xy=np.array([[20., 0.], [20., 0.]]), rss=np.array([[-74, -60, 100], [-70, -62, 100]]))
    q = np.array([[-75, -67, -30], [-60, 100, 100]])
    model = fit_map(h, s, smoothing=0.)
    assert model.visible.tolist() == [True, True, False]
    assert len(model.xy) == 3  # two H positions plus one averaged anchor
    pred = model.predict(q)
    ref = tpm.predict([dict(xy=h[0]["xy"], rss=h[0]["rss"][:, :2])],
                      dict(xy=s["xy"], rss=s["rss"][:, :2]), q[:, :2],
                      None, (40., 1.), (12., -85.), None)
    np.testing.assert_allclose(pred, ref, atol=1e-8, rtol=0)
    q[:, 2] = 100
    np.testing.assert_array_equal(pred, model.predict(q))
    smooth = fit_map(h, s)
    assert np.max(np.abs(smooth.prototypes - model.prototypes)) > 0
    np.testing.assert_array_equal(smooth.prototypes[-1], model.prototypes[-1])
    empty = {key: val[:0] for key, val in s.items()}
    assert len(fit_map(h, empty).xy) == 2
    dark = [{"rss": np.full((2, 3), 100), "xy": np.array([[0, 0], [2, 0]])}]
    np.testing.assert_allclose(fit_map(dark, empty).predict(q), [[1, 0], [1, 0]])
    assert model.predict(q[:0]).shape == (0, 2)
    print("Deployment checks passed")


if __name__ == "__main__":
    main()
