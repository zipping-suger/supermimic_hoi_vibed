import re

import numpy as np

from supermimic_hoi.prepare.dataset import select_clips


def test_selection_covers_every_base_interaction_before_repeating_one():
    names = np.array([f"sub1_box_{base:03}_v{v:02}" for base in range(6) for v in range(5)])
    valid = np.ones(len(names), dtype=bool)
    valid[[0, 1, 2, 3, 4]] = False  # base 000 is corrupted in every variant
    chosen = select_clips(names, valid, 8, np.random.default_rng(0))
    bases = [re.sub(r"_v\d+$", "", names[i]) for i in chosen]
    assert len(set(chosen)) == 8 and "sub1_box_000" not in bases
    assert len(set(bases)) == 5 and max(bases.count(b) for b in set(bases)) == 2
    assert chosen == select_clips(names, valid, 8, np.random.default_rng(0))
