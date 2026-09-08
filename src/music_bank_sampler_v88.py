"""Optional bank/label-balanced TRAIN sampler; does not change live v87."""
from math import lcm

from .music_counterfactual_data_v87 import MusicCounterfactualPairs


def balanced_family_slots(catalog, label):
    music = catalog[catalog.COMPONENT.eq('MUSIC')]
    banks = {value: set(music[music.LABEL.eq(value)].SOURCE_BANK) for value in [0, 1]}
    if not banks[0] or banks[0] != banks[1]:
        raise ValueError('identical nonempty bank support required for both labels')
    families = []
    for bank in sorted(banks[label]):
        subset = music[music.LABEL.eq(label) & music.SOURCE_BANK.eq(bank)]
        families.append([group.index.to_numpy() for _, group in subset.groupby('GENERATOR', dropna=False)])
    # Base sampler selects a slot uniformly, then a source row uniformly. Give
    # every bank the same number of slots, every generator within it equal mass.
    # These are references to index arrays, not duplicate audio/catalog rows.
    slots_per_bank = lcm(*(len(group) for group in families))
    if slots_per_bank * len(families) > 100000:
        raise ValueError('too many exact slots; explicit hierarchical sampling needed')
    return [indices for bank in families for indices in bank for _ in range(slots_per_bank // len(bank))]


class BankBalancedMusicPairs(MusicCounterfactualPairs):
    def __init__(self, catalog, seed, ffmpeg):
        super().__init__(catalog, seed, ffmpeg)
        for label in [0, 1]:
            self.pools['MUSIC', label] = balanced_family_slots(self.catalog, label)
