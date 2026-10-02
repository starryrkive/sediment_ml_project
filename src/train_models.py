import os
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


DATA_FILE = "data/sediment_ml_final.csv"
OUTPUT_DIR = "outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv(DATA_FILE)

print("Dataset shape:", df.shape)

TARGETS = ["d50", "sigma2"]

BASELINE_FEATURES = [
    "temperature",
    "season"
]

EXPANDED_FEATURES = [
    "temperature",
    "season",
    "depth_mean",
    "Hsig",
    "omega"
]

# Same train/test split for every model so the comparison is fair.
train_idx, test_idx = train_test_split(
    np.arange(len(df)),
    test_size=0.30,
    random_state=42
)

train_df = df.iloc[train_idx].copy()
test_df = df.iloc[test_idx].copy()

print("Training rows:", len(train_df))
print("Testing rows :", len(test_df))


def make_preprocessor(features):
    numeric_features = [
        f for f in features
        if f != "season"
    ]

    categorical_features = ["season"]

    return ColumnTransformer(
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


results = []


def evaluate_model(
    feature_set_name,
    features,
    model_name,
    model,
    target
):
    X_train = train_df[features]
    X_test = test_df[features]

    y_train = train_df[target]
    y_test = test_df[target]

    pipeline = Pipeline(
        steps=[
            ("preprocessor", make_preprocessor(features)),
            ("model", model)
        ]
    )

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)

    r2 = r2_score(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(
        mean_squared_error(y_test, predictions)
    )

    results.append({
        "feature_set": feature_set_name,
        "model": model_name,
        "target": target,
        "R2": r2,
        "MAE": mae,
        "RMSE": rmse
    })

    print(
        f"\n{feature_set_name} | "
        f"{model_name} | {target}"
    )
    print(f"R²   : {r2:.4f}")
    print(f"MAE  : {mae:.4f}")
    print(f"RMSE : {rmse:.4f}")


models = {
    "Random Forest": RandomForestRegressor(
        n_estimators=32,
        random_state=42,
        n_jobs=-1
    ),

    "SVR": SVR(
        C=2048,
        epsilon=4
    )
}


for feature_set_name, features in [
    ("Baseline", BASELINE_FEATURES),
    ("Expanded", EXPANDED_FEATURES)
]:

    for target in TARGETS:

        for model_name, model in models.items():

            evaluate_model(
                feature_set_name,
                features,
                model_name,
                model,
                target
            )


results_df = pd.DataFrame(results)

results_file = os.path.join(
    OUTPUT_DIR,
    "model_comparison.csv"
)

results_df.to_csv(
    results_file,
    index=False
)

print("\n==============================")
print("MODEL COMPARISON")
print("==============================")

print(
    results_df.to_string(index=False)
)

print(f"\nSaved to: {results_file}")
