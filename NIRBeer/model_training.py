from typing import Optional, Union
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import GridSearchCV, LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib
from sklearn.compose import TransformedTargetRegressor
import matplotlib.pyplot as plt
import polars as pl
import numpy as np

def train_and_evaluate(
    df: pl.DataFrame,
    holdout_batch: Union[str, list[str]],
    pipeline: Pipeline,
    model_filename_prefix: str,
    separate_models: bool = True,
    param_grid: Optional[dict] = None,
    feature_selector: str = "^WL.*$",
    target_cols: list[str] = ["Ereal_interp", "wtPercEtOH_interp"],
    group_col: str = "BatchName",
    target_names: list[str] = ['Real Extract', 'Ethanol']
) -> list[dict]:
    """
    Trains, evaluates, and plots models based on specified holdout groups.

    Args:
        df (pl.DataFrame): The full dataset.
        holdout_batch (Union[str, list[str]]): A single batch name or a list of names to hold out.
        pipeline (Pipeline): The scikit-learn pipeline to be trained.
        model_filename_prefix (str): A prefix for the saved model files (e.g., 'models/my_model').
                                     The function will append the holdout info.
        separate_models (bool, optional): If True (default) and holdout_batch is a list, a separate
                                          model is trained for each batch. If False, one model is
                                          trained holding out all specified batches combined.
        param_grid (Optional[dict], optional): The hyperparameter grid for GridSearchCV. If None,
                                               GridSearchCV is skipped. Defaults to None.
        feature_selector (str), target_cols (list), group_col (str), target_names (list):
                                               Configuration for data selection and naming.

    Returns:
        list[dict]: A list containing result dictionaries for each model trained.
    """
    # 1. Standardize inputs and set up plots
    holdout_groups = [holdout_batch] if isinstance(holdout_batch, str) else holdout_batch
    n_rows = len(holdout_groups) if separate_models else 1
    
    _, axes = plt.subplots(n_rows, 2, figsize=(12, 5.5 * n_rows), squeeze=False)
    all_results = []

    # 2. Define the core training and evaluation logic as a nested function
    def _run_training(holdout_set, plot_axes_row, model_id):
        # --- Data Splitting ---
        if isinstance(holdout_set, list): # Combined holdout
            df_train = df.filter(pl.col(group_col).is_in(holdout_set).not_())
            df_holdout = df.filter(pl.col(group_col).is_in(holdout_set))
        else: # Single holdout
            df_train = df.filter(pl.col(group_col) != holdout_set)
            df_holdout = df.filter(pl.col(group_col) == holdout_set)

        # --- Feature Extraction ---
        X_train = df_train.select(pl.col(feature_selector)).to_numpy()
        y_train = df_train.select(target_cols).to_numpy()
        groups = df_train[group_col].to_numpy()
        X_holdout = df_holdout.select(pl.col(feature_selector)).to_numpy()
        y_true_holdout = df_holdout.select(target_cols).to_numpy()
        
        # --- Model Setup ---
        model_with_y_scaling = TransformedTargetRegressor(regressor=pipeline, transformer=StandardScaler())
        best_params = "Defaults"
        
        # --- Training ---
        if param_grid:
            print(f"Starting GridSearchCV for holdout '{model_id}' on {len(df_train)} samples...")
            grid_search = GridSearchCV(estimator=model_with_y_scaling, param_grid=param_grid,
                                       cv=LeaveOneGroupOut(), scoring='neg_mean_squared_error', n_jobs=-1)
            grid_search.fit(X_train, y_train, groups=groups)
            best_model = grid_search.best_estimator_
            best_params = grid_search.best_params_
        else:
            print(f"Fitting model for holdout '{model_id}' on {len(df_train)} samples...")
            model_with_y_scaling.fit(X_train, y_train)
            best_model = model_with_y_scaling

        # --- Saving, Evaluation, and Plotting ---
        model_filename = f"{model_filename_prefix}_without_{model_id}.joblib"
        joblib.dump(best_model, model_filename)
        print(f"Model saved to '{model_filename}'")
        y_pred_holdout = best_model.predict(X_holdout)
        metrics = {}
        
        for i, name in enumerate(target_names):
            mse = mean_squared_error(y_true_holdout[:, i], y_pred_holdout[:, i])
            r2 = r2_score(y_true_holdout[:, i], y_pred_holdout[:, i])
            metrics[f'mse_{name.lower().replace(" ", "_")}'] = mse
            metrics[f'r2_{name.lower().replace(" ", "_")}'] = r2
            print(f"Holdout MSE for {name}: {mse:.4f}, $R^2$: {r2:.4f}")
            
            ax = plot_axes_row[i]
            ax.scatter(y_true_holdout[:, i], y_pred_holdout[:, i], alpha=0.7, edgecolors='k', label='Predictions')
            lims = [ax.get_xlim(), ax.get_ylim()]
            lims = [np.min(lims), np.max(lims)]
            ax.plot(lims, lims, 'r--', alpha=0.75, zorder=0, label='Parity Line')
            ax.set_title(f"Holdout: {model_id} - {name}", fontsize=12)
            ax.set_xlabel('True Values'); ax.set_ylabel('Predicted Values')
            ax.grid(True, linestyle='--', alpha=0.6); ax.legend()
            ax.set_aspect('equal', adjustable='box')

        return {'holdout_batch': model_id, 'best_params': best_params, **metrics}

    # 3. Execute the training runs
    if separate_models:
        for i, group_name in enumerate(holdout_groups):
            print(f"\n--- Training Separate Model {i+1}/{len(holdout_groups)} ---")
            result = _run_training(holdout_set=group_name, plot_axes_row=axes[i], model_id=group_name)
            all_results.append(result)
    else:
        print("\n--- Training Single Model on Combined Holdout ---")
        model_id = "Combined"
        result = _run_training(holdout_set=holdout_groups, plot_axes_row=axes[0], model_id=model_id)
        all_results.append(result)

    plt.tight_layout()
    plt.show()
    return all_results