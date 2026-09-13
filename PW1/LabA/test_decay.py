"""
Tests for the decay simulation.

One complete test is given as a model. Add the two tests described in the
lab handout (a negative-rate test, and a test against the analytical law).
Run with:  pytest -v
"""

import numpy as np
import pytest
from decay import simulate, simulate_loop


def test_starts_at_N0():
    # at time zero, no atoms have decayed yet
    assert simulate(1000, 0.4)[0] == 1000


def test_rejects_negative_rate():
    """Verify that calling simulate with a negative lambda raises a ValueError."""
    with pytest.raises(ValueError):
        simulate(1000, -0.4)


def test_matches_law():
    """Verify the average remaining atoms over multiple runs match N0 * exp(-lam * t)."""
    N0 = 1000
    lam = 0.1
    runs = 100

    # Collect the final atom count from multiple simulation runs
    final_counts = [simulate(N0, lam)[-1] for _ in range(runs)]
    avg_final = np.mean(final_counts)

    # Determine total time t from array length (number of steps - 1)
    t = len(simulate(N0, lam)) - 1
    expected_final = N0 * np.exp(-lam * t)

    # Compare average result to physical law with relative tolerance
    assert avg_final == pytest.approx(expected_final, rel=0.1)
