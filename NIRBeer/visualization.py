import matplotlib.pyplot as plt
import polars as pl
import numpy as np
from typing import Any, List

def plot_model_performance(
    model: Any,
    spectra_holdout_df: pl.DataFrame,
    actuals_holdout_df: pl.DataFrame,
    holdout_batch: str,
    target_columns: List[str],
    feature_columns: str = "^WL.*$",
    time_col_spectra: str = "CumulativeTime",
    time_col_actuals: str = "CumulativeTime"
):
    """
    Plots model predictions against actual values using pre-filtered DataFrames.

    Args:
        model (Any): The trained and loaded model object with a .predict() method.
        spectra_holdout_df (pl.DataFrame): DataFrame with spectral data for the holdout set.
        actuals_holdout_df (pl.DataFrame): DataFrame with the ground truth (actual) values.
        feature_columns (List[str]): A list of column names in `spectra_holdout_df` to be used as model features (X).
        target_columns (List[str]): A list of target column names in `actuals_holdout_df`.
        time_col_spectra (str, optional): Name of the time column in `spectra_holdout_df`. Defaults to "CumulativeTime".
        time_col_actuals (str, optional): Name of the time column in `actuals_holdout_df`. Defaults to "CumulativeTime".
    """
    # 1. Extract data from the input DataFrames
    try:
        X_holdout = spectra_holdout_df.filter(pl.col("BatchName") == holdout_batch).select(pl.col(feature_columns)).to_numpy()
        time_X = spectra_holdout_df.filter(pl.col("BatchName") == holdout_batch).get_column(time_col_spectra).to_numpy()

    except Exception as e:
        print(f"❌ Error preparing spectral data: {e}")
        return

    # 2. Get model predictions
    try:
        predicted_values = model.predict(X_holdout)
    except Exception as e:
        print(f"❌ Error during model prediction: {e}")
        return

    # 3. Create subplots
    n_targets = len(target_columns)
    fig, axes = plt.subplots(
        nrows=n_targets, ncols=1,
        sharex=True, squeeze=False
    )
    axes = axes.flatten()
    fig.suptitle("Model Performance on Holdout Data", fontsize=16, y=0.98)

    # 4. Plot each target in its own subplot
    for i, target_name in enumerate(target_columns):
        ax = axes[i]
        
        # Plot predictions
        ax.plot(time_X, predicted_values[:, i], label=f'Predicted {target_name}', marker='.', linestyle='-', markersize=4)

        # Plot actual values
        time_actuals = actuals_holdout_df.filter(pl.col("BatchName") == holdout_batch).get_column(time_col_actuals).to_numpy()
        values_actuals = actuals_holdout_df.filter(pl.col("BatchName") == holdout_batch).get_column(target_name).to_numpy()
        ax.scatter(time_actuals, values_actuals, color='red', label=f'Actual {target_name}', alpha=0.7)

        ax.set_ylabel(target_name)
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.6)

    # 5. Final adjustments and display
    axes[-1].set_xlabel(f"{time_col_actuals} / h")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()