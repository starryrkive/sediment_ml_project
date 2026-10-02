import os
import numpy as np
import pandas as pd
from scipy.io import loadmat


BASE_DIR = "data/raw"
OUTPUT_FILE = "data/sediment_data.csv"

SEASONS = ["Summer", "Winter", "Spring"]


def matlab_time_to_datetime(matlab_time):
    """Convert MATLAB serial date numbers to pandas timestamps."""
    return pd.to_datetime(matlab_time - 719529, unit="D")


def calculate_psd_statistics(psd, diameters):
    """
    Calculate d50 and weighted variance for each LISST measurement.

    psd:
        Particle-size distribution, shape (n_samples, n_bins)

    diameters:
        Diameter of each particle-size bin.
    """

    d50_values = []
    variance_values = []

    for row in psd:

        row = np.asarray(row, dtype=float)

        # Replace invalid values
        row[~np.isfinite(row)] = 0
        row[row < 0] = 0

        total = row.sum()

        if total <= 0:
            d50_values.append(np.nan)
            variance_values.append(np.nan)
            continue

        # Normalize PSD into weights
        weights = row / total

        # Cumulative distribution
        cumulative = np.cumsum(weights)

        # Find d50
        index = np.searchsorted(cumulative, 0.5)

        if index == 0:
            d50 = diameters[0]

        elif index >= len(diameters):
            d50 = diameters[-1]

        else:
            # Linear interpolation between bins
            x1 = cumulative[index - 1]
            x2 = cumulative[index]

            d1 = diameters[index - 1]
            d2 = diameters[index]

            if x2 == x1:
                d50 = d2
            else:
                d50 = d1 + (0.5 - x1) * (d2 - d1) / (x2 - x1)

        # Weighted mean diameter
        weighted_mean = np.sum(weights * diameters)

        # Weighted variance
        variance = np.sum(
            weights * (diameters - weighted_mean) ** 2
        )

        d50_values.append(d50)
        variance_values.append(variance)

    return np.array(d50_values), np.array(variance_values)


def process_season(season):

    print(f"\nProcessing {season}...")

    lisst_path = os.path.join(
        BASE_DIR, season, "lisst.mat"
    )

    ctd_path = os.path.join(
        BASE_DIR, season, "ctd.mat"
    )

    # -------------------------
    # Load LISST
    # -------------------------

    lisst = loadmat(lisst_path)

    lisst_time = lisst["time"].flatten()
    psd = np.asarray(lisst["psd"], dtype=float)
    diameters = lisst["dias"].flatten().astype(float)

    lisst_datetime = matlab_time_to_datetime(lisst_time)

    print("LISST measurements:", len(lisst_datetime))
    print("PSD shape:", psd.shape)

    # Calculate targets
    d50, sigma2 = calculate_psd_statistics(
        psd,
        diameters
    )

    lisst_df = pd.DataFrame({
        "time": lisst_datetime,
        "d50": d50,
        "sigma2": sigma2
    })

    # Hourly aggregation
    lisst_df["hour"] = lisst_df["time"].dt.floor("h")

    lisst_hourly = (
        lisst_df
        .groupby("hour")
        .agg({
            "d50": "median",
            "sigma2": "median"
        })
        .reset_index()
    )

    # -------------------------
    # Load CTD
    # -------------------------

    ctd = loadmat(ctd_path)

    ctd_time = ctd["time"].flatten()
    temperature = ctd["temp"].flatten()

    ctd_datetime = matlab_time_to_datetime(ctd_time)

    ctd_df = pd.DataFrame({
        "time": ctd_datetime,
        "temperature": temperature
    })

    # Hourly temperature
    ctd_df["hour"] = ctd_df["time"].dt.floor("h")

    ctd_hourly = (
        ctd_df
        .groupby("hour")
        .agg({
            "temperature": "mean"
        })
        .reset_index()
    )

    # -------------------------
    # Merge LISST + CTD
    # -------------------------

    merged = pd.merge(
        lisst_hourly,
        ctd_hourly,
        on="hour",
        how="inner"
    )

    merged["season"] = season

    return merged


def main():

    all_data = []

    for season in SEASONS:

        try:
            season_data = process_season(season)
            all_data.append(season_data)

            print(
                f"{season} usable hourly rows:",
                len(season_data)
            )

        except Exception as e:
            print(
                f"ERROR processing {season}: {e}"
            )

    if not all_data:
        raise RuntimeError(
            "No data was successfully processed."
        )

    # Combine seasons
    final_df = pd.concat(
        all_data,
        ignore_index=True
    )

    # Remove missing values
    final_df = final_df.dropna(
        subset=[
            "d50",
            "sigma2",
            "temperature"
        ]
    )

    # Sort chronologically
    final_df = final_df.sort_values(
        "hour"
    ).reset_index(drop=True)

    # Save
    os.makedirs(
        os.path.dirname(OUTPUT_FILE),
        exist_ok=True
    )

    final_df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("\n==============================")
    print("FINAL DATASET")
    print("==============================")

    print("Rows:", len(final_df))
    print("Columns:", list(final_df.columns))

    print("\nMissing values:")
    print(final_df.isna().sum())

    print("\nSummary:")
    print(final_df[
        ["temperature", "d50", "sigma2"]
    ].describe())

    print("\nRows by season:")
    print(final_df["season"].value_counts())

    print(
        f"\nSaved to: {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()
