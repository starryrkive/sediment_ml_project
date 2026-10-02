import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

DATA_FILE = "data/sediment_ml_final.csv"
OUTPUT_DIR = "outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv(DATA_FILE)

FEATURES = [
    "temperature",
    "season",
    "depth_mean",
    "Hsig",
    "omega"
]

TARGETS = ["d50", "sigma2"]

# Same split as the main experiment
train_idx, test_idx = train_test_split(
    np.arange(len(df)),
    test_size=0.30,
    random_state=42
)

train_df = df.iloc[train_idx]

numeric_features = [
    "temperature",
    "depth_mean",
    "Hsig",
    "omega"
]

categorical_features = ["season"]

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            StandardScaler(),
            numeric_features
        ),
        (
            "categorical",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            ),
            categorical_features
        )
    ]
)

for target in TARGETS:

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=32,
                    random_state=42,
                    n_jobs=-1
                )
            )
        ]
    )

    model.fit(
        train_df[FEATURES],
        train_df[target]
    )

    rf = model.named_steps["model"]

    # Get transformed feature names
    feature_names = model.named_steps[
        "preprocessor"
    ].get_feature_names_out()

    importances = rf.feature_importances_

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    })

    # Combine the three season dummy variables into one
    season_mask = importance_df["feature"].str.contains(
        "season_"
    )

    season_importance = importance_df.loc[
        season_mask,
        "importance"
    ].sum()

    importance_df = importance_df.loc[
        ~season_mask
    ].copy()

    importance_df = pd.concat([
        importance_df,
        pd.DataFrame([{
            "feature": "season",
            "importance": season_importance
        }])
    ], ignore_index=True)

    importance_df = importance_df.sort_values(
        "importance",
        ascending=False
    ).reset_index(drop=True)

    print("\n==============================")
    print(f"FEATURE IMPORTANCE: {target}")
    print("==============================")

    print(importance_df.to_string(index=False))

    # Save CSV
    output_csv = os.path.join(
        OUTPUT_DIR,
        f"feature_importance_{target}.csv"
    )

    importance_df.to_csv(
        output_csv,
        index=False
    )

    # Plot
    plt.figure(figsize=(8, 5))

    plt.barh(
        importance_df["feature"][::-1],
        importance_df["importance"][::-1]
    )

    plt.xlabel("Random Forest Feature Importance")
    plt.ylabel("Feature")
    plt.title(
        f"Feature Importance for Predicting {target}"
    )

    plt.tight_layout()

    output_png = os.path.join(
        OUTPUT_DIR,
        f"feature_importance_{target}.png"
    )

    plt.savefig(
        output_png,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    print(f"\nSaved: {output_csv}")
    print(f"Saved: {output_png}")
