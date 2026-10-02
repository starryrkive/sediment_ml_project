import pandas as pd
import os

LISST_FILE = "data/sediment_data.csv"
RBR_FILE = "data/rbr_hourly.csv"
OUTPUT_FILE = "data/sediment_ml_final.csv"

lisst = pd.read_csv(LISST_FILE, parse_dates=["hour"])
rbr = pd.read_csv(RBR_FILE, parse_dates=["hour"])

print("LISST/CTD rows:", len(lisst))
print("RBR rows:", len(rbr))

# Merge observations recorded in the same hour.
merged = pd.merge(
    lisst,
    rbr,
    on=["hour", "season"],
    how="inner"
)

merged = merged.sort_values("hour").reset_index(drop=True)

# Keep only complete ML observations.
feature_columns = [
    "temperature",
    "depth_mean",
    "Hsig",
    "omega",
    "season"
]

target_columns = [
    "d50",
    "sigma2"
]

before = len(merged)

merged = merged.dropna(
    subset=feature_columns + target_columns
)

after = len(merged)

os.makedirs("data", exist_ok=True)
merged.to_csv(OUTPUT_FILE, index=False)

print("\n==============================")
print("FINAL MERGED DATASET")
print("==============================")

print("Rows before dropna:", before)
print("Rows after dropna :", after)
print("Rows lost          :", before - after)

print("\nColumns:")
print(list(merged.columns))

print("\nMissing values:")
print(merged.isna().sum())

print("\nRows by season:")
print(merged["season"].value_counts())

print("\nFeature summary:")
print(
    merged[
        ["temperature", "depth_mean", "Hsig", "omega"]
    ].describe()
)

print("\nTarget summary:")
print(
    merged[
        ["d50", "sigma2"]
    ].describe()
)

print(f"\nSaved to: {OUTPUT_FILE}")
