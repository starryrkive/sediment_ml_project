# ML for Predicting Sediment Particle Size Distributions

## What this project does
This project predicts two continuous sediment properties:
- `d50`: median particle diameter (micrometers)
- `sigma2`: variance of the particle-size distribution (micrometers²)

The reference project describes 8 input features: salinity (`S`), wave-induced velocity (`ub`),
particle index of refraction (`np`), temperature (`T`), `a676/a650`, chlorophyll-a (`chl-a`),
`a450/a676`, and mean tidal velocity (`u`). It uses Random Forest regression and compares it
with Support Vector Regression (SVR).

## IMPORTANT DATA NOTE
The original reference paper describes 1,648 observations, but the original measurement file
is not included in this starter package. The included `data/demo_synthetic_data.csv` is
synthetic and is ONLY for checking that the code runs end-to-end.

Before submission, obtain the actual dataset from your faculty/source and save it as:
`data/sediment_data.csv`

Expected columns:
`S, ub, np, T, a676_a650, chl_a, a450_a676, u, d50, sigma2`

## Setup
```bash
python -m venv .venv
source .venv/bin/activate        # macOS/Linux
# Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run
```bash
python src/train_models.py
```

Or explicitly:
```bash
python src/train_models.py --data data/sediment_data.csv
```

If the real dataset is absent, the script automatically uses the synthetic demo dataset.
The terminal will clearly say `SYNTHETIC DEMO`.

## Outputs
The script creates:
- `outputs/feature_importance.csv`
- `outputs/model_results.csv`
- Random Forest true-vs-predicted plots
- SVR true-vs-predicted plots

## Suggested GitHub structure
```
sediment-ml-project/
├── data/
│   ├── demo_synthetic_data.csv
│   └── sediment_data.csv       # do NOT upload if restricted
├── outputs/
├── src/
│   └── train_models.py
├── README.md
└── requirements.txt
```

## What to say in the demo
1. We load the sediment observations.
2. We select the input features and two targets.
3. Random Forest estimates feature importance.
4. We keep the four features identified as most important in the reference project.
5. We train Random Forest and SVR regression models.
6. We compare predicted values with measured values using R², MAE and RMSE.
7. We visualize true vs predicted values.
