from dataclasses import dataclass

@dataclass
class Config:
    # Outlier detection
    outlier_alpha: float = 0.01
    pca_var_threshold: float = 0.99  # retain components up to this cumulative variance

    # Derivative-based sampling
    derivative_bins: int = 10
    total_sample_cap: int = 3000  # total selected spectra across all batches
    min_per_batch: int = 100      # ensure coverage even in slow batches
    seed: int = 42

    # Grid search / CV
    max_pls_components: int = 20
    scoring: str = "r2"  # multi-output r2
    n_jobs: int = -1
    verbose: int = 1

    # Preprocessing candidates (will be pruned by availability at runtime)
    # Savitzky-Golay parameter grids (window must be odd and <= n_features)
    savgol_windows: tuple = (9, 11, 15, 21, 25)
    savgol_polyorders: tuple = (2, 3)
    savgol_derivatives: tuple = (0, 1, 2)

    # Whittaker smoothing parameter grid (lambda controls smoothness)
    whittaker_lambdas: tuple = (10, 100, 1000, 10000)
    whittaker_d: tuple = (2,)