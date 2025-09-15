import numpy as np
import polars as pl
from dataclasses import dataclass
from typing import Optional
from sklearn.decomposition import PCA
from scipy.stats import f as f_dist
from scipy.stats import norm
from chemotools.scatter import StandardNormalVariate


@dataclass
class OutlierResults:
    t2: np.ndarray
    q: np.ndarray
    t2_limit: float
    q_limit: float
    inliers_mask: np.ndarray
    num_inliers: int
    n_components: int
    df_inliers: pl.DataFrame


def _jackson_mudholkar_q_limit(residual_eigs: np.ndarray, alpha: float) -> float:
    # residual_eigs are eigenvalues (variances) of the residual subspace (excluded PCs)
    theta1 = np.sum(residual_eigs)
    theta2 = np.sum(residual_eigs**2)
    theta3 = np.sum(residual_eigs**3)
    if theta1 <= 0 or theta2 <= 0:
        return 0.0
    h0 = 1.0 - (2.0 * theta1 * theta3) / (3.0 * (theta2**2))
    h0 = max(h0, 1e-6)
    ca = norm.ppf(1 - alpha)
    term = (ca * np.sqrt(2.0 * theta2) * h0 / theta1) + 1.0 + (theta2 * h0 * (h0 - 1.0)) / (theta1**2)
    q_limit = theta1 * (term ** (1.0 / h0))
    return float(q_limit)

def _hotellings_t2_limit(n: int, k: int, alpha: float) -> float:
    if k >= n:
        # Degenerate; return very large threshold
        return np.inf
    fcrit = f_dist.ppf(1 - alpha, dfn=k, dfd=n - k)
    return (k * (n - 1) / (n - k)) * fcrit

def detect_outliers(
    df_nir: pl.DataFrame, 
    alpha: float = 0.01, 
    var_threshold: float = 0.99,
    range_cut: Optional[object] = None
) -> OutlierResults:
    """
    Detect outliers in NIR spectra data using PCA-based Hotelling's T2 and Q-residuals.
    
    Parameters
    ----------
    df_nir : pl.DataFrame
        Polars DataFrame containing NIR spectra with columns starting with "WL"
    alpha : float, optional
        Significance level for outlier detection, by default 0.01
    var_threshold : float, optional
        Variance threshold for PCA component selection, by default 0.99
    range_cut : Optional[object], optional
        RangeCut object for trimming spectral edges, by default None
    
    Returns
    -------
    OutlierResults
        Dataclass containing outlier detection results
    """
    # Extract wavelength columns using regex pattern
    wavelength_cols = df_nir.select(pl.col("^WL.*$")).columns
    X = df_nir.select(wavelength_cols).to_numpy()
    
    # Apply RangeCut if provided
    if range_cut is not None:
        X = range_cut.fit_transform(X)
    
    # Apply Robust Normal Variate preprocessing
    rnv = StandardNormalVariate()
    X_corrected = rnv.fit_transform(X)

    n, p = X_corrected.shape

    # Fit full PCA to obtain all eigenvalues
    pca_full = PCA(svd_solver="full")
    scores_full = pca_full.fit_transform(X_corrected)
    eigs_full = pca_full.explained_variance_
    cumvar = np.cumsum(pca_full.explained_variance_ratio_)
    k = int(np.searchsorted(cumvar, var_threshold) + 1)
    k = max(2, min(k, min(n - 1, p)))  # guardrails

    # Scores for retained components
    Tk = scores_full[:, :k]
    lambdas_k = eigs_full[:k]

    # Hotelling's T2
    t2 = np.sum((Tk**2) / lambdas_k, axis=1)
    t2_lim = _hotellings_t2_limit(n=n, k=k, alpha=alpha)

    # Q-residuals (SPE)
    # Reconstruct using only first k components: zero-out trailing scores
    scores_kpad = np.zeros_like(scores_full)
    scores_kpad[:, :k] = Tk
    Xhat = (scores_kpad @ pca_full.components_) + pca_full.mean_
    residuals = X_corrected - Xhat
    q = np.sum(residuals**2, axis=1)

    # Q limit per Jackson–Mudholkar using residual eigs
    residual_eigs = eigs_full[k:]
    q_lim = _jackson_mudholkar_q_limit(residual_eigs=residual_eigs, alpha=alpha)
    inliers = (t2 <= t2_lim) & (q <= q_lim)
    num_inliers = np.sum(inliers)

    df_inliers = df_nir.filter(pl.Series(inliers))

    return OutlierResults(
        t2=t2,
        q=q,
        t2_limit=float(t2_lim),
        q_limit=float(q_lim),
        inliers_mask=inliers,
        num_inliers=int(num_inliers),
        n_components=k,
        df_inliers=df_inliers
    )