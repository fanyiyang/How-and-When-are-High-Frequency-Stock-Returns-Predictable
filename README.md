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
- `MatchingEngine.py` — limit-order-book matching engine that reconstructs trades and L1+ snapshots from raw Shenzhen Stock Exchange tick data. Used by `load_data_and_build_feature.py`.
- `load_data_and_build_feature.py` — load raw data and build features/labels.
- `train.py` — train models and generate results.
- `docs/steps.md` — detailed step-by-step explanation of each stage.

## Matching engine
`MatchingEngine.py` replays the exchange's order and trade feeds to rebuild the
limit order book and execution stream. It supports the normal continuous-auction
rule, the ChiNext "freeze" (鸽笼) price-cage rule, and the opening call auction.
It expects per-day tick files (`am_/pm_hq_order_spot.csv`, `am_/pm_hq_trade_spot.csv`,
`am_/pm_snap_level_spot.csv`, `am_hq_snap_spot.csv`) under
`<file_path>/<stock>.XSHE/<year>/<MMDD>/`. Supply your own tick data in this layout;
none is bundled. See the `__main__` block in the file for a runnable example.

## Detailed documentation
See `docs/steps.md` for a full walkthrough of each step, inputs/outputs, and the training loop logic.
