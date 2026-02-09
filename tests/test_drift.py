from __future__ import annotations

import numpy as np

from fraud_mlops.drift import population_stability_index


def test_psi_detects_shift() -> None:
    rng = np.random.default_rng(7)
    expected = rng.normal(0, 1, size=5000)
    actual_stable = rng.normal(0, 1, size=5000)
    actual_shifted = rng.normal(1.2, 1.2, size=5000)

    stable_psi = population_stability_index(expected, actual_stable)
    shifted_psi = population_stability_index(expected, actual_shifted)

    assert shifted_psi > stable_psi

