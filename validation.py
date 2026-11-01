import pandas as pd
FEATURES = ["age", "hours-per-week", "education", "occupation"]
def validate_batch(frame):
    missing=set(FEATURES)-set(frame.columns)
    if missing: raise ValueError(f"Missing columns: {sorted(missing)}")
    if not 1 <= len(frame) <= 10000: raise ValueError("Supply 1–10000 rows")
    frame=frame[FEATURES].copy()
    for name,low,high in [("age",18,90),("hours-per-week",1,99)]:
        frame[name]=pd.to_numeric(frame[name],errors="raise")
        if not frame[name].between(low,high).all() or (frame[name] % 1 != 0).any():
            raise ValueError(f"{name} must be a whole number between {low} and {high}")
    for name in ["education","occupation"]:
        if frame[name].isna().any(): raise ValueError(f"{name} must not be missing")
        frame[name]=frame[name].astype(str).str.strip()
        if frame[name].eq("").any() or frame[name].str.len().gt(100).any(): raise ValueError(f"Invalid {name}")
    return frame
