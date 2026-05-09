import numpy as np
from scipy.stats import wasserstein_distance


def sliced_wasserstein(x: np.ndarray, y: np.ndarray,
                       n_slices: int = 200, seed: int = 0) -> float:
    """
    Sliced Wasserstein distance between two 2D point clouds.

    Projects both sets onto n_slices random unit vectors and averages
    the 1D Wasserstein distance over all projections.

    Args:
        x        : (N, D) samples from distribution P
        y        : (M, D) samples from distribution Q
        n_slices : number of random projection directions
        seed     : RNG seed for reproducibility

    Returns:
        scalar sliced Wasserstein distance
    """
    rng = np.random.default_rng(seed)
    D = x.shape[1]
    directions = rng.standard_normal((n_slices, D))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)

    total = 0.0
    for d in directions:
        total += wasserstein_distance(x @ d, y @ d)

    return total / n_slices
