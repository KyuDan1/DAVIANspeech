"""Fixed per-file component composition; no fitted router or global statistics."""
import numpy as np


def compose_components(xlsr, eat, spear=None):
    arrays = [np.asarray(xlsr, np.float64), np.asarray(eat, np.float64)]
    if spear is not None:
        arrays.append(np.asarray(spear, np.float64))
    if any(a.ndim != 2 or a.shape[1] != 5 or a.shape != arrays[0].shape for a in arrays):
        raise ValueError('aligned arrays [files, 5] required')
    if any(not np.isfinite(a).all() or np.any((a < 0) | (a > 1)) for a in arrays):
        raise ValueError('finite probabilities in [0, 1] required')
    voice = arrays[0][:, 1].copy()
    if spear is not None:
        first = np.clip(voice, 1e-6, 1 - 1e-6)
        second = np.clip(arrays[2][:, 1], 1e-6, 1 - 1e-6)
        logits = .5 * (np.log(first) - np.log1p(-first)) + .5 * (np.log(second) - np.log1p(-second))
        voice = 1 / (1 + np.exp(-logits))
    music, vp, mp = arrays[1][:, 2], arrays[1][:, 3], arrays[1][:, 4]
    # Conditional fake probabilities remain un-gated in the component columns.
    # Only file risk includes presence. Noisy-OR is an explicit independence
    # approximation, not an assertion that component events are independent.
    file_fake = 1 - (1 - voice * vp) * (1 - music * mp)
    return np.stack([file_fake, voice, music, vp, mp], -1)
