import matplotlib.pyplot as plt
import re
import polars as pl
import numpy as np
from NIRBeer.model_training import train_and_evaluate
from sklearn.cross_decomposition import PLSRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from chemotools.scatter import ExtendedMultiplicativeScatterCorrection
from chemotools.derivative import SavitzkyGolay
from NIRBeer.feature_selection import VIPSelector
from NIRBeer.outlier_detection import detect_outliers
from chemotools.feature_selection import RangeCut


plt.style.use('default')
plt.rcParams.update({
    'font.size': 12,             
    'axes.labelsize': 12,        
    'axes.titlesize': 14,        
    'xtick.labelsize': 12,       
    'ytick.labelsize': 12,       
    'legend.fontsize': 12,       
    'figure.titlesize': 14       
})

true_batch_names = {
    "01_Zwickelbier": "1. Zwickelbier",
    "02_Zwickelbier": "2. Zwickelbier",
    "03_Zwickelbier": "3. Zwickelbier",
    "04_Wiener_Lager": "Vienna Lager",
    "05_European_Red_Lager": "European Red Lager",
    "06_Uni_Pils": "Pilsner"
}



## Load Data

df_dsa = pl.read_ipc("data/Beer_Analyzer_Data_full.feather")
df_nir = pl.read_ipc("data/Fermentation_NIR_absorbance_full.feather")

# Ensure 'BatchName' is of string type
df_dsa = df_dsa.with_columns(
    pl.col('BatchName').cast(pl.Utf8)
)

# Reverse absorbance columns to match wavelength order (wavenumber to wavelength conversion)
regex_pattern = r"^WL.*$"
cols_to_reverse = [col for col in df_nir.columns if re.match(regex_pattern, col)]

reversal_map = dict(zip(cols_to_reverse, cols_to_reverse[::-1]))

df_nir = df_nir.select(
    [
        pl.col(reversal_map[col]).alias(col) if col in reversal_map
        else pl.col(col)
        for col in df_nir.columns
    ]
)

# Find index of 1200 nm and 1800 nm
wavelengths = 1e7 / np.linspace(3857,9994,359)
# reverse wavelengths to match wavenumber order
wavelengths = wavelengths[::-1]

index_upper= np.argmin(np.abs(wavelengths - 1850))
index_lower = np.argmin(np.abs(wavelengths - 1200))



preprocessing = Pipeline([("range_cut", RangeCut(index_lower, index_upper))])

result = detect_outliers(df_nir, alpha=0.05, var_threshold=0.95, preprocessing=preprocessing)

print("Removed", df_nir.shape[0] - result.num_inliers, "outliers")


from NIRBeer.interpolation import interpolate_targets_to_spectra
from NIRBeer.sampling import adaptive_time_sampler

df_interpolate = interpolate_targets_to_spectra(result.df_inliers, df_dsa)

min_time = 3 
max_time = 16
k_rate = 50*3/2
exponent = 1

selected_df = df_interpolate.group_by("BatchName", maintain_order=True).map_groups(
    lambda df_group: adaptive_time_sampler(
        df_group,
        min_dt=min_time,
        max_dt=max_time,
        rate_k=k_rate,
        exp_val=exponent
    )
)

holdout_batch = ["05_European_Red_Lager", "06_Uni_Pils"]

pipeline = Pipeline([
        ('rangecut', RangeCut(index_lower, index_upper)),
        ('emsc', ExtendedMultiplicativeScatterCorrection(order=2)),
        ('sg', SavitzkyGolay(polynomial_order=2, window_size=11, derivate_order=1)),
        ('scaler', StandardScaler()),
        ('vip', VIPSelector(n_components_pls=6, vip_threshold=0.95)),
        ('pls', PLSRegression(n_components=6, scale=False))
    ])

train_and_evaluate(selected_df, holdout_batch, pipeline, f"models/PLS_final", separate_models=True, true_batch_names=true_batch_names, plot_output_filename="figures/PLS_final.png")

# 0.0788 0.0324 0.1121 0.0513
# 0.0950 0.0348 0.0850 0.0641