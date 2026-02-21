import numpy as np
from src.eigen import solve_eigen

def test_small_grid():
    vals, _ = solve_eigen(N=5, potential='well', n_eigs=3)
    assert len(vals) == 3
    # Basic check: eigenvalues should be ascending
    assert np.all(np.diff(vals) >= 0), "Eigenvalues are not sorted"

def test_medium_grid():
    vals, _ = solve_eigen(N=20, potential='harmonic', n_eigs=5)
    assert len(vals) == 5
    # Basic check: eigenvalues should be ascending
    assert np.all(np.diff(vals) >= 0), "Eigenvalues are not sorted"