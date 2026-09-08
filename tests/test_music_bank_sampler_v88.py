from collections import Counter
from fractions import Fraction
import unittest
import pandas as pd
from src.music_bank_sampler_v88 import balanced_family_slots


class BankSamplingTest(unittest.TestCase):
    def test_bank_mass_equal_despite_label_and_generator_imbalance(self):
        rows = []
        for label in [0, 1]:
            for bank, generators, count in [('A', 1 if label == 0 else 2, 7), ('B', 1 if label == 0 else 3, 2)]:
                rows.extend(dict(COMPONENT='MUSIC', LABEL=label, SOURCE_BANK=bank, GENERATOR=str(g))
                            for g in range(generators) for _ in range(count))
        frame = pd.DataFrame(rows)
        for label in [0, 1]:
            slots = balanced_family_slots(frame, label)
            bank_mass = Counter()
            generator_mass = Counter()
            for slot in slots:
                for index in slot:
                    row = frame.iloc[index]
                    mass = Fraction(1, len(slots) * len(slot))
                    self.assertEqual(row.LABEL, label)
                    bank_mass[row.SOURCE_BANK] += mass
                    generator_mass[row.SOURCE_BANK, row.GENERATOR] += mass
            self.assertEqual(bank_mass, {'A': Fraction(1, 2), 'B': Fraction(1, 2)})
            for bank in ['A', 'B']:
                self.assertEqual(len({v for (b, _), v in generator_mass.items() if b == bank}), 1)

    def test_no_silent_dropping_unpaired_banks(self):
        frame = pd.DataFrame([dict(COMPONENT='MUSIC', LABEL=i, SOURCE_BANK=str(i), GENERATOR='g') for i in [0, 1]])
        with self.assertRaises(ValueError):
            balanced_family_slots(frame, 0)


if __name__ == '__main__':
    unittest.main()
