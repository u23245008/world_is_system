"""Run the pinned Gentle pipeline followed by a configurable voice-band boost."""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
from . import branch
from . import voice_balance

GENTLE = branch.BASE_EXTRA | branch.METHODS["gentle"]
RAW3 = branch.METHODS["raw3"]["trial_post"].copy()


def pilot(noisy, sr, noise_refiner=None, noise_reference=None):
    branch.EXTRA = GENTLE if noise_refiner is None else GENTLE | {"noise_refiner": noise_refiner}
    return branch.denoise_channel(noisy, sr, noise_reference)


def apply(gentle, noisy, sr, config):
    if len(noisy) < 2048 or not np.any(noisy):
        return noisy.copy()
    level_db = config.get("level_db", 0.)
    boost = voice_balance.refine(gentle, sr, RAW3 | config, noisy=noisy)
    if level_db:
        boost *= 10**(level_db/20)
        peak = float(np.max(np.abs(boost)))
        if peak > .999:
            boost *= .999/peak
    return boost


def denoise(noisy, sr, config=None, noise_refiner=None, noise_reference=None):
    x = np.asarray(noisy, dtype=np.float64)
    if x.ndim != 1 or not np.isfinite(x).all() or sr < 8000:
        raise ValueError("Expected finite mono audio at 8 kHz or above")
    if config is None:
        config = json.loads((HERE/"next_recommended.json").read_text())
    return apply(pilot(x, sr, noise_refiner, noise_reference), x, sr, config)
