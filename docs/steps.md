# Detailed steps

This document explains what each core script does, what it reads, and what it writes.

## 1. `load_data_and_build_feature.py`
This script has two stages: **raw data loading** and **feature/label construction**.

### 1.1 Raw data loading (`load_raw_data`)
**Purpose:** Merge per-day matching results into one year-level raw dataset.  
**Flow:**
1. Iterate over daily files under `../../data/HFData/HS300_data/<stock>.XSHE/2020/`.
2. Run `MatchingEngine.Engine` for each day (`trade_before` rule).
3. Collect `execute_total_list` and concatenate into a single DataFrame.
4. Write `/data/work/yangsq/dataset_all/<stock>fulldata.csv`.

**Output:**
```
/data/work/yangsq/dataset_all/<stock>fulldata.csv
```

### 1.2 Feature/label construction (`build_feature_set`)
**Purpose:** Build microstructure predictors and return targets, then write the training set.  
**Flow:**
1. Read `fulldata.csv`, set `TradingDay` as the index, and limit to 2020.
2. Load monthly turnover files (`/data/work/yangsq/数据/2020*.csv`) for `Turnover`.
3. **Response variables (labels):**
   - 5s/30s returns, 10/200 trade returns, 1000/20000 volume returns.
4. **Predictors (features):**
   - 13 microstructure signals (Breadth, Immediacy, VolumeAll, VolumeAvg, Lambda,
     LobImbalance, TxnImbalance, PastReturn, Turnover, AutoCov, QuotedSpread,
     EffectiveSpread).
   - Computed over 9 windows in calendar/transaction/volume time.
5. Group by trading day, compute features/labels, and concatenate into a training set.
6. Write `/data/work/yangsq/trainset_all/<stock>trainset.csv`.

**Output:**
```
/data/work/yangsq/trainset_all/<stock>trainset.csv
```

## 2. `train.py`
This script handles model training, hyperparameter tuning, and outputs.

### 2.1 Read and preprocess
1. Read daily training files from `/data/work/yangsq/trainset_all/<stock>/calendar/`.
2. Concatenate into a single DataFrame indexed by trading day.
3. Clean column names to use only letters/numbers/underscores.
4. Use the first 108 columns as `X`, and columns 109/110 as `y1/y2`.

### 2.2 Hyperparameter search
Every 20 trading days, run a grid search for Lasso / RF / Ridge / LightGBM:
1. Use Optuna’s `GridSampler`.
2. Evaluate each parameter with three rolling 5-day train/test splits.

### 2.3 Rolling prediction
Inside each 20-day cycle, repeat:
1. Train on the previous 5 days.
2. Test on the next day (out-of-sample).
3. Record R² for Lasso / RF / Ridge / LGBM.
4. Save feature importance for Lasso/RF/LGBM (index uses `i+j-20`).

### 2.4 Outputs
After training completes, write:
```
/data/work/yangsq/Rset_all/<stock>Rset.csv
/data/work/yangsq/Hyperset_all/<stock>Hyperset.csv
/data/work/yangsq/Feature_importance_all/<model>/<stock>Feature_importance.csv
/data/work/yangsq/Return_all/<stock>Value.pickle
```

## 3. How to run (two core scripts)
```
python load_data_and_build_feature.py
python train.py
```
`stockname_list` defaults to a single stock for quick iteration; add more codes to batch runs.
