from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cross_decomposition import PLSRegression
import numpy as np


def _calculate_vip_scores(pls_model):
    """
    Calculates VIP scores from a fitted scikit-learn PLS model.
    This version is corrected to avoid the NumPy deprecation warning.
    """
    if not hasattr(pls_model, 'x_scores_'):
        raise ValueError("The PLS model must be fitted before calculating VIP scores.")

    t = pls_model.x_scores_      # T scores (n_samples, n_components)
    w = pls_model.x_weights_     # W weights (n_features, n_components)
    q = pls_model.y_loadings_    # Q loadings (n_targets, n_components)

    p, h = w.shape  # p: number of features, h: number of components

    # Sum of squares of explained variance for each component
    ssy = np.diag(t.T @ t @ q.T @ q).reshape(h, -1)
    ssy_total = np.sum(ssy)

    vips = np.zeros((p,))
    for i in range(p):
        # Weight of the i-th variable for each component
        weight = np.array([(w[i, j] / np.linalg.norm(w[:, j]))**2 for j in range(h)])
        
        # Corrected calculation: Use np.dot on 1D arrays to ensure a scalar result
        vips[i] = np.sqrt(p * np.dot(ssy.flatten(), weight) / ssy_total)

    return vips

class VIPSelector(BaseEstimator, TransformerMixin):
    """
    A transformer that selects features based on VIP scores from an internal PLS model.
    The VIP threshold is a tunable hyperparameter.
    """
    def __init__(self, n_components_pls=10, vip_threshold=1.0):
        self.n_components_pls = n_components_pls
        self.vip_threshold = vip_threshold
        self.mask_ = None
        self.vip_scores_ = None

    def fit(self, X, y):
        # Fit an internal PLS model to calculate VIP scores
        internal_pls = PLSRegression(n_components=self.n_components_pls)
        internal_pls.fit(X, y)

        # Calculate VIP scores and create the selection mask
        self.vip_scores_ = _calculate_vip_scores(internal_pls)
        self.mask_ = self.vip_scores_ >= self.vip_threshold
        
        # Check if any features were selected
        if not np.any(self.mask_):
            print(f"Warning: No features selected with VIP threshold {self.vip_threshold}. Keeping all features.")
            self.mask_ = np.ones(X.shape[1], dtype=bool)

        return self

    def transform(self, X):
        return X[:, self.mask_]