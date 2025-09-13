import numpy as np
import polars as pl
from scipy.interpolate import PchipInterpolator

def interpolate_targets_to_spectra(
    spectra_df: pl.DataFrame,
    targets_df: pl.DataFrame,
    ensure_monotonic: bool = True,
    extrapolate: bool = True
) -> pl.DataFrame:
    
    # Create result dataframe with additional columns
    result_dfs = []
    
    for bname in spectra_df["BatchName"].unique():
        df_spec = spectra_df.filter(pl.col("BatchName") == bname)
        df_tgt = targets_df.filter(pl.col("BatchName") == bname)
        
        if df_tgt.height < 2:
            # Not enough targets for interpolation, keep original data
            result_dfs.append(df_spec.with_columns([
                pl.lit(np.nan).alias("Ereal_interp"),
                pl.lit(np.nan).alias("wtPercEthanol_interp"),
                pl.lit(np.nan).alias("dE_dt"),
                pl.lit(np.nan).alias("dEtOH_dt")
            ]))
            continue

        # Use existing CumulativeTime column directly
        t_spec = df_spec["CumulativeTime"].to_numpy(dtype=float)
        t_tgt = df_tgt["CumulativeTime"].to_numpy(dtype=float)
        yE = df_tgt["Ereal"].to_numpy(dtype=float)
        yEt = df_tgt["wtPercEthanol"].to_numpy(dtype=float)

        if ensure_monotonic:
            # Deduplicate and enforce strictly increasing x for PCHIP
            ord_idx = np.argsort(t_tgt)
            t_tgt = t_tgt[ord_idx]
            yE = yE[ord_idx]
            yEt = yEt[ord_idx]
            # Remove duplicates (keep first occurrence)
            uniq_mask = np.concatenate(([True], np.diff(t_tgt) > 0))
            t_tgt = t_tgt[uniq_mask]
            yE = yE[uniq_mask]
            yEt = yEt[uniq_mask]

        # Build PCHIP interpolators
        fE = PchipInterpolator(t_tgt, yE, extrapolate=extrapolate)
        fEt = PchipInterpolator(t_tgt, yEt, extrapolate=extrapolate)
        dE = fE.derivative()
        dEt = fEt.derivative()

        E_vals = fE(t_spec)
        Et_vals = fEt(t_spec)
        dE_vals = dE(t_spec)
        dEt_vals = dEt(t_spec)

        # Add interpolated values to the spectra dataframe for this batch
        result_df = df_spec.with_columns([
            pl.Series("Ereal_interp", E_vals),
            pl.Series("wtPercEthanol_interp", Et_vals),
            pl.Series("dE_dt", dE_vals),
            pl.Series("dEtOH_dt", dEt_vals)
        ])
        
        result_dfs.append(result_df)

    # Combine all batch results
    out = pl.concat(result_dfs)
    
    # Combined derivative magnitude (L2 norm) for sampling
    out = out.with_columns(
        np.sqrt(pl.col("dE_dt").to_numpy(dtype=float)**2 +
                pl.col("dEtOH_dt").to_numpy(dtype=float)**2).alias("deriv_mag")
    )
    
    return out
    