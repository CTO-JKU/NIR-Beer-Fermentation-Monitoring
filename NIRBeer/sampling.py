import polars as pl
import numpy as np


def adaptive_time_sampler(
    df_group: pl.DataFrame,
    min_dt: float,
    max_dt: float,
    rate_k: float,
    exp_val: float
) -> pl.DataFrame:
    """
    Applies the adaptive sampling logic to a single batch (a DataFrame group).

    This function is designed to be used with Polars' group_by().apply().
    """
    last_time = -float('inf')
    selected_rows = []

    # iter_rows is efficient for this kind of stateful iteration
    for row in df_group.iter_rows(named=True):
        t = row['CumulativeTime']
        dE = row['derivative_magnitude']

        # Calculate the desired local time step (dt_local)
        # The core of the algorithm: dt is small when dE is large, and large when dE is small.
        dt_local = max_dt * np.exp(-rate_k * dE) ** exp_val

        # Enforce the minimum time step
        dt_local = max(dt_local, min_dt)

        if t >= last_time + dt_local:
            selected_rows.append(row)
            last_time = t

    if not selected_rows:
        return df_group.clear() # Return empty frame if no rows selected

    # Return a new DataFrame constructed from the selected rows
    return pl.from_dicts(selected_rows, schema=df_group.schema).sort(["BatchName", "CumulativeTime"])