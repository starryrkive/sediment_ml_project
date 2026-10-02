import os
import glob
import numpy as np
import pandas as pd
from scipy.io import loadmat

BASE_DIR = "data/raw/RBR"
OUTPUT_FILE = "data/rbr_hourly.csv"

SEASONS = ["Summer", "Winter", "Spring"]

# Conservative quality-control threshold.
# This is a project preprocessing assumption, not a value specified
# by the original dataset documentation.
MAX_HSIG = 2.0


def matlab_time_to_datetime(matlab_time):
    return pd.to_datetime(matlab_time - 719529, unit="D")


def process_file(path, season):
    try:
        m = loadmat(path)

        tstart = float(np.asarray(m["tstart"]).squeeze())

        depth = np.asarray(m["depth"], dtype=float).flatten()
        depth = depth[np.isfinite(depth)]

        Hsig = float(np.asarray(m["Hsig"]).squeeze())
        omega = float(np.asarray(m["omega"]).squeeze())

        # Quality control
        if len(depth) == 0:
            return None

        if not np.isfinite(Hsig) or Hsig <= 0 or Hsig > MAX_HSIG:
            return None

        if not np.isfinite(omega) or omega <= 0:
            return None

        depth_mean = np.mean(depth)

        timestamp = matlab_time_to_datetime(tstart)

        return {
            "hour": timestamp.floor("h"),
            "depth_mean": depth_mean,
            "Hsig": Hsig,
            "omega": omega,
            "season": season,
        }

    except Exception as e:
        print(f"Error processing {path}: {e}")
        return None


def process_season(season):
    pattern = os.path.join(BASE_DIR, season, "*.mat")
    files = glob.glob(pattern)

    print(f"\nProcessing {season}...")
    print("Files found:", len(files))

    rows = []

    for i, path in enumerate(files, start=1):
        result = process_file(path, season)

        if result is not None:
            rows.append(result)

        if i % 500 == 0:
            print(f"Processed {i}/{len(files)}")

    df = pd.DataFrame(rows)

    if df.empty:
        raise RuntimeError(f"No usable RBR data found for {season}")

    # Multiple bursts can occur in the same hour.
    # Median provides robustness against remaining burst-level noise.
    hourly = (
        df.groupby(["hour", "season"])
        .agg({
            "depth_mean": "median",
            "Hsig": "median",
            "omega": "median",
        })
        .reset_index()
    )

    print("Usable burst records:", len(df))
    print("Rejected burst records:", len(files) - len(df))
    print("Hourly records:", len(hourly))

    return hourly


def main():
    all_data = []

    for season in SEASONS:
        season_data = process_season(season)
        all_data.append(season_data)

    final_df = pd.concat(all_data, ignore_index=True)

    final_df = final_df.sort_values("hour").reset_index(drop=True)

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    final_df.to_csv(OUTPUT_FILE, index=False)

    print("\n==============================")
    print("RBR HOURLY DATASET")
    print("==============================")
    print("Rows:", len(final_df))
    print("Columns:", list(final_df.columns))

    print("\nMissing values:")
    print(final_df.isna().sum())

    print("\nSummary:")
    print(final_df[["depth_mean", "Hsig", "omega"]].describe())

    print("\nRows by season:")
    print(final_df["season"].value_counts())

    print(f"\nSaved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
