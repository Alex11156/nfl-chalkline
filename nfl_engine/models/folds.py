"""Walk-forward backtesting, shared by the game and player models.

For each test season Y: fit on every earlier season, predict Y. Nothing from
Y or later reaches the model that predicts Y, which is the only honest way to
score a forecast, and it is how the models are used in real life.

A finished fold gives the same answer every time until its data, inputs or
settings change. With `cache=...` each fold's predictions are stored under a
fingerprint of exactly those things (models_store/wf_cache/), so the weekly
retrain recomputes only what moved: normally the current season's fold.
"""
import hashlib

import numpy as np
import pandas as pd

from nfl_engine.config import MODELS_STORE

CACHE = MODELS_STORE / "wf_cache"


def _digest(*parts) -> str:
    h = hashlib.sha1()
    for p in parts:
        h.update(p if isinstance(p, bytes) else repr(p).encode())
    return h.hexdigest()[:20]


def walk_forward(d: pd.DataFrame, cols: list[str], target: str, seasons: list[int],
                 fit_predict, cache: tuple | None = None) -> np.ndarray:
    """Out-of-sample predictions for every row of `d`, in d's order (NaN for
    rows outside `seasons`).

    fit_predict(train, test, season) -> predictions for the rows of `test`.
    cache = (name, settings) turns the fold cache on; `settings` must hold
    everything other than the data that decides the result (model parameters,
    library versions).
    """
    season = d["season"].to_numpy()
    out = np.full(len(d), np.nan)
    if cache is not None:
        name, settings = cache
        # one order-sensitive digest of each season's inputs and outcomes
        rows = pd.util.hash_pandas_object(d[cols + [target]], index=False).to_numpy()
        digests = {int(s): _digest(rows[season == s].tobytes()) for s in np.unique(season)}
    for y in seasons:
        test = season == y
        if not test.any():
            continue
        file = None
        if cache is not None:
            key = _digest(settings, cols, target, [v for s, v in digests.items() if s <= y])
            file = CACHE / f"{name}_{y}_{key}.npy"
            if file.exists():
                out[test] = np.load(file)
                continue
        out[test] = fit_predict(d[season < y], d[test], y)
        if file is not None:
            CACHE.mkdir(parents=True, exist_ok=True)
            for old in CACHE.glob(f"{name}_{y}_*.npy"):
                old.unlink()
            np.save(file, out[test])
    return out
