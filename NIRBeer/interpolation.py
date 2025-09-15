import numpy as np
import polars as pl
from scipy.interpolate import PchipInterpolator

def interpolate_targets_to_spectra(
    spectra_df: pl.DataFrame,
    targets_df: pl.DataFrame,
) -> pl.DataFrame:
    
    # Create result dataframe with additional columns
    result_dfs = []
    
    for bname in spectra_df["BatchName"].unique().sort().to_list():
        df_spec = spectra_df.filter(pl.col("BatchName") == bname)
        df_tgt = targets_df.filter(pl.col("BatchName") == bname)
    
        # Use existing CumulativeTime column directly
        t_spec = df_spec["CumulativeTime"].to_numpy()
        t_tgt = df_tgt["CumulativeTime"].to_numpy()
        yE = df_tgt["Ereal"].to_numpy()
        yEt = df_tgt["wtPercEtOH"].to_numpy()

        # Build PCHIP interpolators
        fE = PchipInterpolator(t_tgt, yE)
        fEt = PchipInterpolator(t_tgt, yEt)
        dE = fE.derivative()
        dEt = fEt.derivative()

        E_vals = fE(t_spec)
        Et_vals = fEt(t_spec)
        dE_vals = dE(t_spec)
        dEt_vals = dEt(t_spec)

        # Add interpolated values to the spectra dataframe for this batch
        result_df = df_spec.with_columns([
            pl.Series("Ereal_interp", E_vals),
            pl.Series("wtPercEtOH_interp", Et_vals),
            pl.Series("dE_dt", dE_vals),
            pl.Series("dEtOH_dt", dEt_vals)
        ])
        
        result_dfs.append(result_df)

    # Combine all batch results
    out = pl.concat(result_dfs)
    
    # Combined derivative magnitude (L2 norm) for sampling
    out = out.with_columns(
        np.sqrt(pl.col("dE_dt")**2 + pl.col("dEtOH_dt")**2).alias("derivative_magnitude")
    )
    return out
    
if __name__ == "__main__":  
   
    df_dsa = pl.read_ipc("data/Beer_Analyzer_Data_full.feather")
    df_nir = pl.read_ipc("data/Fermentation_NIR_absorbance_full.feather")

    result = interpolate_targets_to_spectra(df_nir, df_dsa)

    print(result)