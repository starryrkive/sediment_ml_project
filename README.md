# Machine Learning for Predicting Sediment Particle Size Distributions

## Project Overview

This project uses machine learning to predict sediment particle-size distribution properties from environmental and hydrodynamic measurements collected in South San Francisco Bay.

The two prediction targets are:

- **d50** — median sediment particle diameter (µm)
- **sigma2** — variance of the particle-size distribution (µm²)

The project is based on the Stanford reference study *Machine learning for predicting sediment particle size distributions* by Galen Egan.

The original study used Random Forest (RF) regression and Support Vector Regression (SVR) to predict sediment particle-size properties from environmental, biological, and hydrodynamic variables.

This implementation uses the publicly available South San Francisco Bay field dataset and focuses on variables that could be extracted and reliably aligned from the available LISST, CTD, and RBR measurements.

---

## Dataset

The source data are from three field campaigns in South San Francisco Bay:

- **Summer:** July–August 2018
- **Winter:** January–February 2019
- **Spring:** April–May 2019

Source dataset:

https://purl.stanford.edu/wv787xr0534

The dataset contains measurements from instruments including LISST, CTD, RBR pressure loggers, Vectrino instruments, ADVs, and ADCPs.

### Data used in this project

The LISST measurements provide particle-size distributions. These are processed to calculate:

- `d50` — median particle diameter
- `sigma2` — variance of the particle-size distribution

CTD measurements provide:

- `temperature`

RBR measurements at platform P2 provide:

- `depth_mean`
- `Hsig`
- `omega`

The LISST and CTD measurements are aggregated hourly and merged with the hourly RBR measurements.

After preprocessing and merging:

- **1,764 hourly observations**
- **8 columns**
- **0 missing values**

Season distribution:

| Season | Observations |
|---|---:|
| Winter | 637 |
| Summer | 592 |
| Spring | 535 |

---

## Features and Targets

### Input features

| Feature | Description |
|---|---|
| `temperature` | Hourly mean water temperature |
| `season` | Summer, Winter, or Spring |
| `depth_mean` | Mean depth within an RBR burst, aggregated hourly |
| `Hsig` | Significant wave height |
| `omega` | RBR wave-related angular frequency |

### Prediction targets

| Target | Description |
|---|---|
| `d50` | Median particle diameter (µm) |
| `sigma2` | Variance of particle-size distribution (µm²) |

---

## Project Pipeline

The project follows this workflow:

```text
Raw MATLAB (.mat) files
        |
        v
LISST + CTD preprocessing
        |
        v
Hourly sediment dataset
        |
        +------------------+
        |                  |
        v                  v
      LISST              CTD
   particle size      temperature
        |                  |
        +--------+---------+
                 |
                 v
        RBR preprocessing
                 |
                 v
        Hourly merged dataset
                 |
                 v
       Model training/testing
                 |
          +------+------+
          |             |
          v             v
     Random Forest     SVR
          |             |
          +------+------+
                 |
                 v
       Model comparison
                 |
                 v
       Feature importance
```

---

## Models

Two regression models are compared.

### 1. Random Forest Regression

A Random Forest combines multiple decision trees and averages their predictions.

Configuration:

- `n_estimators = 32`
- `random_state = 42`
- `n_jobs = -1`

### 2. Support Vector Regression

SVR is used as a second regression approach for comparison.

Configuration:

- `C = 2048`
- `epsilon = 4`

Both models use the same train/test split:

- **70% training**
- **30% testing**
- `random_state = 42`

Numeric features are standardized for SVR, and the categorical `season` variable is one-hot encoded.

---

## Experimental Comparison

Two feature sets are evaluated.

### Baseline

Uses only:

```text
temperature
season
```

### Expanded

Uses:

```text
temperature
season
depth_mean
Hsig
omega
```

This allows us to examine whether adding hydrodynamic measurements improves prediction compared with a simpler environmental baseline.

---

## Results

### d50 prediction

| Model | Feature Set | R² | MAE (µm) | RMSE (µm) |
|---|---|---:|---:|---:|
| Random Forest | Baseline | 0.176 | 25.78 | 34.22 |
| SVR | Baseline | 0.393 | 22.92 | 29.37 |
| Random Forest | Expanded | **0.564** | **17.91** | **24.89** |
| SVR | Expanded | 0.477 | 19.19 | 27.25 |

### sigma2 prediction

| Model | Feature Set | R² | MAE | RMSE |
|---|---|---:|---:|---:|
| Random Forest | Baseline | 0.038 | 681.54 | 912.93 |
| SVR | Baseline | 0.269 | 592.49 | 795.68 |
| Random Forest | Expanded | **0.453** | **475.89** | **688.71** |
| SVR | Expanded | 0.418 | 509.03 | 710.39 |

The expanded feature set improves the test performance of both prediction targets compared with the corresponding baseline models.

---

## Feature Importance

Random Forest feature importance was calculated for the expanded models.

### d50

| Feature | Importance |
|---|---:|
| temperature | 0.418 |
| Hsig | 0.235 |
| depth_mean | 0.159 |
| omega | 0.095 |
| season | 0.093 |

### sigma2

| Feature | Importance |
|---|---:|
| temperature | 0.377 |
| Hsig | 0.182 |
| depth_mean | 0.172 |
| season | 0.171 |
| omega | 0.099 |

These values describe the relative contribution of the features within the fitted Random Forest models. They should not be interpreted as causal relationships.

---

## Reference Study Comparison

The reference Stanford project used a richer set of environmental, biological, and hydrodynamic variables, including:

- salinity (`S`)
- wave-induced velocity (`ub`)
- particle index of refraction (`np`)
- temperature (`T`)
- `a676/a650`
- chlorophyll-a
- `a450/a676`
- mean tidal velocity (`u`)

The reference study reported approximately:

- RF d50: R² = 0.81
- RF sigma²: R² = 0.75
- SVR d50: R² = 0.76
- SVR sigma²: R² = 0.76

The present implementation should **not be interpreted as a direct reproduction** of those results because the available extracted features and preprocessing pipeline differ from the reference study.

The reference study used 1,648 examples, while this project produces 1,764 hourly observations after merging LISST, CTD, and RBR measurements.

---

## Repository Structure

```text
sediment_ml_project/
│
├── data/
│   ├── sediment_data.csv
│   ├── rbr_hourly.csv
│   └── sediment_ml_final.csv
│
├── outputs/
│   ├── model_comparison.csv
│   ├── feature_importance_d50.csv
│   ├── feature_importance_sigma2.csv
│   ├── feature_importance_d50.png
│   ├── feature_importance_sigma2.png
│   ├── rf_d50_true_vs_pred.png
│   ├── rf_sigma2_true_vs_pred.png
│   ├── svr_d50_true_vs_pred.png
│   └── svr_sigma2_true_vs_pred.png
│
├── src/
│   ├── preprocess_real_data.py
│   ├── preprocess_rbr.py
│   ├── merge_datasets.py
│   ├── train_models.py
│   └── feature_importance.py
│
├── requirements.txt
├── README.md
└── .gitignore
```

Raw `.mat` files are intentionally excluded from GitHub because of their size.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/starryrkive/sediment_ml_project.git
cd sediment_ml_project
```

Create a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Running the Project

The processed dataset is already included in the repository.

### Train and compare models

```bash
python src/train_models.py
```

This generates:

```text
outputs/model_comparison.csv
outputs/rf_d50_true_vs_pred.png
outputs/rf_sigma2_true_vs_pred.png
outputs/svr_d50_true_vs_pred.png
outputs/svr_sigma2_true_vs_pred.png
```

### Generate feature importance

```bash
python src/feature_importance.py
```

This generates:

```text
outputs/feature_importance_d50.csv
outputs/feature_importance_sigma2.csv
outputs/feature_importance_d50.png
outputs/feature_importance_sigma2.png
```

---

## Reproducing the Preprocessing

To reproduce the complete preprocessing pipeline from the original MATLAB files, obtain the Stanford source dataset and place the relevant files under:

```text
data/raw/
├── Summer/
│   ├── lisst.mat
│   └── ctd.mat
├── Winter/
│   ├── lisst.mat
│   └── ctd.mat
├── Spring/
│   ├── lisst.mat
│   └── ctd.mat
└── RBR/
    ├── Summer/
    │   └── P2/
    ├── Winter/
    │   └── P2/
    └── Spring/
        └── P2/
```

Then run:

```bash
python src/preprocess_real_data.py
python src/preprocess_rbr.py
python src/merge_datasets.py
python src/train_models.py
python src/feature_importance.py
```

The preprocessing scripts use hourly aggregation to align measurements from the different instruments.

---

## Data Quality Handling

The raw RBR files contain some anomalous `Hsig` values. A project-level quality-control filter was applied:

```text
0 < Hsig <= 2 m
```

This rejected 19 out of 8,549 RBR burst files, approximately 0.22% of the files.

The final hourly RBR dataset uses median aggregation across bursts within each hour.

The LISST pressure field was not used because the dataset documentation indicates that the LISST pressure sensor was broken.

---

## Limitations

This is a one-week educational ML project rather than a production scientific model.

Important limitations include:

1. The available feature set differs from the original reference study.
2. The biological/optical variables from the reference study were not included in the final model.
3. The current evaluation uses a random 70/30 train/test split.
4. Performance may depend on the specific deployment periods represented in the dataset.
5. Feature importance indicates model association, not causation.
6. The model has not been independently validated on a completely separate field campaign.

Future work could investigate time-aware validation, cross-season validation, additional biological/optical features, hyperparameter tuning, and generalization to other locations.

---

## References

Egan, G.  
*Machine learning for predicting sediment particle size distributions.*  
Stanford project report.

South San Francisco Bay field dataset:  
https://purl.stanford.edu/wv787xr0534
