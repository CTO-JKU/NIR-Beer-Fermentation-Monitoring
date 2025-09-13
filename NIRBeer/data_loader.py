import pandas as pd

def load_spectra(feather_path: str) -> pd.DataFrame:
    df = pd.read_feather(feather_path)
    # Normalize column names
    df.columns = [str(c) for c in df.columns]
    # Parse timestamp
    if "TimeStamp" not in df.columns or "BatchName" not in df.columns:
        raise ValueError("Spectra file must contain 'TimeStamp' and 'BatchName'.")
    
    df["TimeStamp"] = pd.to_datetime(df["TimeStamp"], utc=True, errors="coerce")
    if df["TimeStamp"].isna().any():
        raise ValueError("Invalid timestamps in spectra file.")
    # Sort
    df = df.sort_values(["BatchName", "TimeStamp"]).reset_index(drop=True)
    return df

def load_targets(feather_path: str) -> pd.DataFrame:
    df = pd.read_feather(feather_path)
    df.columns = [str(c) for c in df.columns]
    if "TimeStamp" not in df.columns or "Batchname" not in df.columns:
        # Accept either BatchName or Batchname
        if "BatchName" in df.columns:
            df = df.rename(columns={"BatchName": "Batchname"})
        else:
            raise ValueError("Target file must contain 'TimeStamp' and 'Batchname'.")
    if "Ereal" not in df.columns or "wtPercEthanol" not in df.columns:
        raise ValueError("Target file must contain 'Ereal' and 'wtPercEthanol'.")
    df["TimeStamp"] = pd.to_datetime(df["TimeStamp"], utc=True, errors="coerce")
    if df["TimeStamp"].isna().any():
        raise ValueError("Invalid timestamps in target file.")
    # Normalize naming to BatchName across project
    df = df.rename(columns={"Batchname": "BatchName"})
    df = df.sort_values(["BatchName", "TimeStamp"]).reset_index(drop=True)
    return df

def get_wavelength_columns(df: pd.DataFrame) -> list[str]:
    # All columns starting with WL (WL1 ... WL359)
    wl_cols = [c for c in df.columns if c.startswith("WL")]
    if not wl_cols:
        raise ValueError("No wavelength columns found (expected names like 'WL1', 'WL2', ...).")
    return wl_cols