import matplotlib.pyplot as plt
import polars as pl
import numpy as np
from typing import Any, List
from matplotlib.colors import Normalize
import matplotlib.cm as cm
import numpy as np
import polars as pl
from typing import List, Optional, Any
from chemotools.feature_selection import RangeCut
from sklearn.pipeline import Pipeline


def plot_spectra_by_batch(
    df: pl.DataFrame,
    wavelengths: np.ndarray,
    batch_names: Optional[List[str]] = None,
    n_cols: int = 3,
    preprocessing: Optional[Pipeline] = None,
    cmap_name: str = 'viridis',
    output_filename: Optional[str] = None
) -> None:
    """
    Plots NIR spectra for different batches from a Polars DataFrame, with spectra colored by time.

    Args:
        df (pl.DataFrame): DataFrame containing the spectral data. Must include columns
                           'BatchName', 'CumulativeTime', and spectral columns (e.g., 'WL_...').
        batch_names (List[str]): A list of titles for each batch plot. Must match the
                                 number of unique batches.
        wavelengths (np.ndarray): Array of the full wavelength values for the x-axis before cutting.
        n_cols (int, optional): Number of columns in the subplot grid. Defaults to 3.
        preprocessing (Any, optional): A preprocessing transformer with a .fit_transform() method.
                                       If None, no preprocessing is applied. Defaults to None.
        cmap_name (str, optional): Name of the Matplotlib colormap. Defaults to 'viridis'.
        output_filename (Optional[str], optional): If provided, the plot is saved to this
                                                   file path. Defaults to None.
    """
    # --- 1. Data and Plot Setup ---
    # Get unique sorted batch identifiers from the dataframe
    batches = df["BatchName"].unique().sort().to_list()
    n_batches = len(batches)

    if n_batches != len(batch_names):
        raise ValueError("The length of `batch_names` must match the number of unique batches in the DataFrame.")

    # Create subplot grid
    n_rows = int(np.ceil(n_batches / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows), sharex=True, sharey=True)
    axes = np.array(axes).flatten() # Flatten for easy iteration

    # Get colormap
    cmap = cm.get_cmap(cmap_name)

    # **Crucially, normalize color based on the min/max time across the ENTIRE dataset**
    # This ensures a consistent color scale for all subplots.
    all_time_values = df.select(pl.col("CumulativeTime")).to_numpy().flatten()
    norm = Normalize(vmin=all_time_values.min(), vmax=all_time_values.max())

    if preprocessing is not None:
        for _, step in preprocessing.steps:
            if isinstance(step, RangeCut):
                wavelengths = wavelengths[step.start:step.end]
                break

    # --- 3. Plotting Loop ---
    for i, batch in enumerate(batches):
        ax = axes[i]
        
        # Filter data for the current batch
        batch_data = df.filter(pl.col("BatchName") == batch)
        spectra = batch_data.select(pl.col("^WL.*$")).to_numpy()
        time_values = batch_data.select(pl.col("CumulativeTime")).to_numpy().flatten()
        
        # Apply the range cut to the spectra
        spectra_cut = preprocessing.fit_transform(spectra)
        
        # Plot each spectrum with its corresponding color based on time
        for j, spectrum in enumerate(spectra_cut):
            color = cmap(norm(time_values[j]))
            ax.plot(wavelengths, spectrum, color=color, alpha=0.2)
        
        ax.set_title(batch_names[i])
        ax.set_xlabel("Wavelength / nm")
        ax.set_ylabel("Absorbance / AU")
    
    # Hide any unused subplots
    for i in range(n_batches, len(axes)):
        axes[i].set_visible(False)

    # --- 4. Final Touches (Colorbar and Layout) ---
    # Add a single, horizontal colorbar for the entire figure
    cbar_ax = fig.add_axes([0.15, 0.06, 0.7, 0.02])  # [left, bottom, width, height]
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
    cbar.set_label('Time / h')

    # Adjust layout to prevent overlap and make space for the colorbar
    plt.tight_layout(rect=[0, 0.08, 1, 1])

    # Save the figure if a filename is provided
    if output_filename:
        plt.savefig(output_filename, dpi=600, bbox_inches='tight')
    
    plt.show()



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