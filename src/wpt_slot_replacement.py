"""Replace one known logit-mixture slot, never add a second copy of that expert."""
import numpy as np

EPS = 1e-5


def logit(values):
    values = np.clip(np.asarray(values, dtype=np.float64), EPS, 1 - EPS)
    return np.log(values) - np.log1p(-values)


def replace_wpt_slot(final_probability, old_expert, new_expert, weight=.4):
    arrays = [np.asarray(value, dtype=np.float64) for value in (final_probability, old_expert, new_expert)]
    if len({a.shape for a in arrays}) != 1:
        raise ValueError('all probabilities must have the same shape')
    if not 0 < weight < 1:
        raise ValueError('a recoverable slot requires weight strictly between 0 and 1')
    if any(not np.isfinite(a).all() or ((a < 0) | (a > 1)).any() for a in arrays):
        raise ValueError('finite probabilities in [0,1] required')
    final, old, new = arrays
    base_logit = (logit(final) - weight * logit(old)) / (1 - weight)
    bound = float(logit(1 - EPS))
    if np.max(np.abs(base_logit), initial=0.) > bound + 1e-4:
        raise ValueError('old expert/anchor incompatible with the declared convex slot')
    result = np.exp(-np.logaddexp(0, -(logit(final) + weight * (logit(new) - logit(old)))))
    # Preserve exact anchor values for the identity-replacement ablation.
    return np.where(new == old, final, result)
