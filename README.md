# How and When Are High-Frequency Stock Returns Predictable?

This repository contains a compact, runnable replication of the paper **“How and When Are High-Frequency Stock Returns Predictable?”** focused on building microstructure features from high-frequency data and training predictive models on those features.

## Quickstart
1. Install dependencies:
   ```bash
   pip install pandas numpy scikit-learn optuna lightgbm xgboost
   ```
2. Check paths in the scripts to match your data layout:
   - `load_data_and_build_feature.py`
   - `train.py`
3. Run the two core steps:
   ```bash
   python load_data_and_build_feature.py
   python train.py
   ```

## Project layout (minimal files)
- `load_data_and_build_feature.py` — load raw data and build features/labels.
- `train.py` — train models and generate results.
- `docs/steps.md` — detailed step-by-step explanation of each stage.

## Detailed documentation
See `docs/steps.md` for a full walkthrough of each step, inputs/outputs, and the training loop logic.
