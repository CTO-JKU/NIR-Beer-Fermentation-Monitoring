import matplotlib.pyplot as plt
import polars as pl
import numpy as np
from typing import Any, List
from matplotlib.colors import Normalize
import matplotlib.cm as cm
import numpy as np
import polars as pl
from typing import List, Optional, Any, Dict
from sklearn.pipeline import Pipeline


def plot_spectra_by_batch(
    df: pl.DataFrame,
    wavelengths: np.ndarray,
    batch_names: List[str],
    n_cols: int = 2,
    preprocessing: Optional[Pipeline] = None,
    cmap_name: str = 'plasma',
    output_filename: Optional[str] = None,
    fig_title: Optional[str] = None,
    row_spacing: float = 0.3  # NEW: vertical spacing between rows (hspace)
) -> None:
    """
    Plots NIR spectra with per-row y-axis scaling and improved layout.

    Args:
        ...
        row_spacing (float): vertical spacing (hspace) between subplot rows.
    """
    # --- 1. Data Validation and Preparation ---
    batches = df["BatchName"].unique().sort().to_list()
    n_batches = len(batches)
    if batch_names and n_batches != len(batch_names):
        raise ValueError("Length of `batch_names` must match the number of unique batches.")

    processed_wavelengths = wavelengths
    if preprocessing:
        for name, step in preprocessing.steps:
            if hasattr(step, 'start') and hasattr(step, 'end'): # Safer check
                processed_wavelengths = wavelengths[step.start:step.end]
                break

    # --- 2. Figure and Colormap Setup ---
    n_rows = (n_batches + n_cols - 1) // n_cols
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(5 * n_cols, 4 * n_rows),
        sharex=False,
        sharey=False,
        squeeze=False
    )
    axes = axes.flatten()

    # Apply extra vertical spacing between rows
    fig.subplots_adjust(hspace=row_spacing, top=0.95, bottom=0.12)

    if fig_title:
        fig.suptitle(fig_title, fontweight='bold')

    cmap = cm.get_cmap(cmap_name)
    norm = Normalize(vmin=df["CumulativeTime"].min(), vmax=df["CumulativeTime"].max())

    # --- 3. Plotting Loop ---
    for i, (batch_id, batch_title) in enumerate(zip(batches, batch_names)):
        ax = axes[i]
        batch_df = df.filter(pl.col("BatchName") == batch_id)
        spectra = batch_df.select(pl.col("^WL.*$")).to_numpy()
        time_values = batch_df["CumulativeTime"].to_numpy()
        processed_spectra = preprocessing.fit_transform(spectra) if preprocessing else spectra

        for j in range(processed_spectra.shape[0]):
            color = cmap(norm(time_values[j]))
            ax.plot(processed_wavelengths, processed_spectra[j], color=color, alpha=0.4, linewidth=1.5)

        ax.set_title(batch_title)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.set_ylabel("Absorbance / AU")
        ax.set_xlabel('Wavelength / nm')
        ax.tick_params(axis='both', which='both')
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_visible(True)

    # --- 4. Unify Y-Axis Limits Per Row ---
    for row in range(n_rows):
        row_axes = axes[row * n_cols : (row + 1) * n_cols]
        min_vals, max_vals = [], []
        for ax in row_axes:
            if ax.get_visible() and ax.has_data():
                min_vals.append(ax.get_ylim()[0])
                max_vals.append(ax.get_ylim()[1])

        if not min_vals: continue
        row_ymin, row_ymax = min(min_vals), max(max_vals)
        for ax in row_axes:
            ax.set_ylim(row_ymin, row_ymax)

    # --- 5. Final Touches (Labels, Colorbar, Layout) ---
    for i in range(n_batches, len(axes)):
        axes[i].set_visible(False)

    # Colorbar
    cbar_ax = fig.add_axes([0.2, 0.03, 0.6, 0.015])
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal', pad=0.1)
    cbar.set_label('Time / h')

    if output_filename:
        plt.savefig(output_filename, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_filename}")
    plt.show()




import matplotlib.pyplot as plt
import numpy as np
import warnings

def plot_detected_outliers(t2, q, inliers_mask, t2_limit, q_limit,
                                    fig_size=(8.0, 6.0), output_filename: Optional[str] = None, dpi=300):
    """
    Generates an advanced scatter plot of Hotelling's T^2 vs. Q residuals.

    This improved version features:
    - Semantic coloring (blue for inliers, red for outliers).
    - A logarithmic scale to better visualize data clustered near the origin.
    - Labeled quadrants for clear interpretation.

    Args:
        t2 (np.ndarray): Array of Hotelling's T^2 values.
        q (np.ndarray): Array of Q residual values.
        inliers_mask (np.ndarray): Boolean mask where True indicates an inlier.
        t2_limit (float): The threshold for Hotelling's T^2.
        q_limit (float): The threshold for Q residuals.
        save_path (str, optional): Path to save the figure. Defaults to None.
        show_plot (bool, optional): Whether to display the plot. Defaults to True.
        dpi (int, optional): Dots per inch for the saved figure. Defaults to 300.
    """
    # Ensure data is positive for log scale.
    # A small positive constant is added to prevent log(0) errors.
    if np.any(t2 <= 0) or np.any(q <= 0):
        warnings.warn("T^2 or Q values contain non-positive values. "
                      "These points will be omitted from the log-scale plot.")
    
    positive_mask = (t2 > 0) & (q > 0)
    t2_plot = t2[positive_mask]
    q_plot = q[positive_mask]
    inliers_mask_plot = inliers_mask[positive_mask]

    # --- Plot Setup ---
    fig, ax = plt.subplots(figsize=fig_size)
    
    inliers = inliers_mask_plot
    outliers = ~inliers_mask_plot
    
    # 1. Semantic Coloring: Plot inliers and outliers as separate series
    ax.scatter(t2_plot[inliers], q_plot[inliers], 
               c='cornflowerblue', alpha=0.7, s=50, label='Inliers')
    ax.scatter(t2_plot[outliers], q_plot[outliers], 
               c='crimson', alpha=0.8, s=60, marker='x', label='Outliers')

    # --- Scales and Limits ---
    # 2. Use Log Scale instead of an inset plot
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    # Add limit lines
    ax.axvline(t2_limit, color="black", ls="--", lw=1.5)
    ax.axhline(q_limit, color="black", ls="--", lw=1.5)

    # --- Annotations and Labels ---
    ax.set_xlabel("Hotelling's $T^2$", fontsize=12)
    ax.set_ylabel("$Q$ residuals", fontsize=12)
    
    # 3. Add Explanatory Quadrant Labels
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # Add text labels for the limit lines
    ax.text(t2_limit * 1.1, ylim[1] * 0.9 , f' $T^2$ limit {t2_limit:.2f}', 
            ha='left', va='top', rotation=90, fontsize=9)
    ax.text(xlim[1] * 0.9, q_limit * 1.1, f' $Q$ residual limit = {q_limit:.2f}', 
            ha='right', va='bottom', fontsize=9)
            
    # Add counts textbox
    outliers_count = np.sum(~inliers_mask)
    inliers_count = len(inliers_mask) - outliers_count
    textstr = f'Inliers: {inliers_count}\nOutliers: {outliers_count}'
    props = dict(boxstyle='round', facecolor='white', alpha=0.8)
    ax.text(xlim[0] * 1.3, ylim[1] * 0.7, textstr, fontsize=10,
            va='top', ha='left', bbox=props)

    # --- Final Touches ---
    ax.legend(loc='lower right', edgecolor='black')
    ax.grid(True, which="both", ls="--", alpha=0.3)
    fig.tight_layout()

    if output_filename:
        plt.savefig(output_filename, dpi=dpi, bbox_inches='tight')
        print(f"Plot saved to {output_filename}")
   
    plt.show()



def plot_interpolation(
    df_interp: pl.DataFrame,
    df_actual: pl.DataFrame,
    target_col: str,
    target_ylabel: str,
    batch_name_map: Dict[str, str], # Added parameter for the name mapping
    n_cols: int = 2,
    fig_title: Optional[str] = None,
    row_spacing=0.35,
    col_spacing=0.2
) -> None:
    """
    Generates a grid of plots for batch process data, using Polars DataFrames.

    Args:
        df_interp (pl.DataFrame): Polars DataFrame with the interpolated data.
        df_actual (pl.DataFrame): Polars DataFrame with the actual data.
        target_col (str): The base name of the column to plot.
        target_ylabel (str): The y-axis label for each subplot.
        batch_name_map (Dict[str, str]): Dictionary to map BatchName to a display title.
        n_cols (int): The number of columns for the subplot grid.
        fig_title (Optional[str]): The main title for the entire figure.
    """
    # --- 1. Get unique batch names using Polars ---
    batches = df_actual["BatchName"].unique().sort().to_list()
    n_batches = len(batches)
    if n_batches == 0:
        print("DataFrame contains no batches to plot.")
        return

    # --- 2. Prepare plot layout ---
    n_rows = int(np.ceil(n_batches / n_cols))

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(5 * n_cols, 4.5 * n_rows), 
        sharex=True, sharey=True,
        squeeze=False
    )
    axes = axes.flatten()

    plt.subplots_adjust(hspace=row_spacing, wspace=col_spacing)

    # --- 3. Plotting Loop using Polars filtering ---
    for i, batch_name in enumerate(batches):
        ax = axes[i]
        
        # Filter data using the native Polars .filter() method
        batch_interp_data = df_interp.filter(pl.col("BatchName") == batch_name)
        batch_actual_data = df_actual.filter(pl.col("BatchName") == batch_name)

        ax.scatter(
            batch_interp_data["CumulativeTime"],
            batch_interp_data[f"{target_col}_interp"],
            s=5,
            color='cornflowerblue',
            label='Interpolated samples'
        )
        
        ax.scatter(
            batch_actual_data["CumulativeTime"],
            batch_actual_data[target_col],
            label='Actual samples',
            color='crimson',
            alpha=0.5
        )
        
        # Map batch names to titles using the provided dictionary
        plot_title = batch_name_map.get(batch_name, batch_name)
        ax.set_title(plot_title)
        
        # Set individual labels for each subplot
        ax.set_xlabel("Time / h")
        ax.set_ylabel(target_ylabel)
        ax.tick_params(axis='x', labelbottom=True)
        ax.tick_params(axis='y', labelleft=True)
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        ax.legend()

    # --- 4. Clean up and finalize the plot ---
    # Hide any unused subplots
    for i in range(n_batches, len(axes)):
        axes[i].set_visible(False)
    
    if fig_title:
        fig.suptitle(fig_title)

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