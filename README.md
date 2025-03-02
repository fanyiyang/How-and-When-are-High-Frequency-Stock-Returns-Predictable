
## Project Overview  
**How and When are High-Frequency Stock Returns Predictable?** is a research project exploring the predictability of ultra high-frequency stock returns (and the time between trades, or **durations**) using detailed market data and machine learning techniques ([EconPapers: How and When are High-Frequency Stock Returns Predictable?](https://econpapers.repec.org/RePEc:nbr:nberwo:30366#:~:text=Abstract%3A%20This%20paper%20studies%20the,data%20on%20a%20scale%20of)).This repository provides the code to reproduce the findings that **short-horizon stock returns can be predicted with meaningful accuracy using microstructure data**, identifying *when* (under what market conditions and timeframes) and *how* (with which features and models) such predictability arises.


## Installation Instructions  
To set up the environment and dependencies for this project, follow these steps:
- **Install Python Packages**: The code is written in Python and relies on several libraries. You can install the required packages using pip. For example:  
  ```bash
  pip install pandas numpy scikit-learn optuna lightgbm xgboost
  ```  
  This will install:  
  * **pandas** (for data manipulation)  
  * **numpy** (for numerical computations)  
  * **scikit-learn** (for machine learning models like Random Forests, Lasso, Ridge, MLP)
  * **optuna** (for hyperparameter tuning)   
  * **lightgbm** (LightGBM gradient boosting model) 
  * **xgboost** (XGBoost gradient boosting model) 
  Ensure these are installed before running the code. You may use a virtual environment or conda environment to manage dependencies.  
- **Prepare Data**: The high-frequency dataset is not included due to size and licensing constraints. You will need tick-by-tick trade and quote data for the relevant stocks. In the original study, the authors used the complete trades and quotes for 101 large-cap U.S. stocks (S&P 100 constituents) from Jan 2019 to Dec 2020. In this replication, the code expects data for stocks in the HS300 index (likely year 2020). Obtain the data from a market data provider or exchange, ensuring it includes every trade and quote update with timestamps. Once obtained, organize the data as expected by the code (see **Dataset Information** below for details).  
- **File Path Configuration**: Update file paths in the code if necessary. The code references directories like `../../data/HFData/HS300_data/` and `/data/work/yangsq/trainset_all/` for input/output. You may need to modify these paths to point to where your data is stored. For example, in `PreTrain.py` and other scripts, adjust the base directory to match your system’s file structure.  

After installing the libraries and setting up the data files, you should be ready to run the analysis code.

## Usage Guide  
This section explains how to use the provided code to replicate the analyses and results. The repository includes several Python scripts that correspond to different stages of the experiment:

1. **Data Preprocessing (Feature Construction)** – **`PreTrain.py`**:  
   This script reads raw tick data and constructs the predictor features and response variables. It implements the methodology for computing various microstructure features from the trade and quote data. For each stock (and each trading day), it calculates a wide range of predictors (see **Methodology** for details) and the corresponding short-horizon outcomes (future return and duration). Before running this, ensure your data files are in the expected folder structure. You might need to edit `PreTrain.py` to specify which stocks to process. By default, the code uses a list of representative stocks (e.g., 12 stock codes) for an initial pretraining step. Running this script will produce intermediate files (e.g., daily feature data stored as pickles in a `trainset_all/<stock>/` directory). To execute the script, run:  
   ```bash
   python PreTrain.py
   ```  
   *This will calculate all the predictor variables and save the processed dataset for the next steps.*  
2. **Model Training and Evaluation (Calendar Time)** – **`Train copy calendar.py`**:  
   This script trains prediction models on the constructed dataset using calendar time splitting (i.e., real time intervals). It loads the processed data (from the output of PreTrain), and then for each stock and each day, it builds models to predict short-term returns and durations. The script tries multiple machine learning models: Random Forest, Lasso regression, Ridge regression, LightGBM, XGBoost, and a Neural Network (MLP). It uses `optuna` to tune hyperparameters for these models (with functions like `RFscore`, `Lassoscore`, etc., for Optuna trials). The training procedure appears to use rolling or block-wise train-test splits of days (e.g., training on a set of days and testing on the next day in sequence). It also records feature importance for the tree-based models and coefficients for linear models, which helps identify which predictors are most useful. To use this script, you may need to edit it to specify which stock’s data to train on (e.g., set the `stockname` variable or run a loop for multiple stocks). Then run:  
   ```bash
   python "Train copy calendar.py"
   ```  
   *(Note: the file name contains spaces, so quoting it or escaping spaces is necessary in the command.)* This will output model performance metrics (such as R² or classification accuracy) and possibly save results like feature importance. Check the console output and any files the script writes for the results on each stock.  
3. **Model Training and Evaluation (Event Time / Intraday Periodicity)** – **`periodic train.py`**:  
   This script likely complements the previous one by performing similar training and analysis in *transaction time* or focusing on intraday periods. The original study examined predictability on a **per-transaction basis** in addition to real time. The `periodic train.py` script probably loads the same processed data but might slice it by fixed numbers of trades or by intraday segments to analyze “periodic” effects. Running this script (`python "periodic train.py"`) will perform the analyses for these alternate setups. Again, ensure any required parameters (like stock name or data paths) are configured inside the script before execution.  
4. **Interpreting Output**: After running the above scripts, you will have results that can be compared to the findings of the paper. Look for printed output or log files that show things like prediction accuracy for returns, prediction accuracy for trade direction, R² for duration predictions, feature importance rankings, etc. These will allow you to verify the key outcomes – for example, how accuracy declines as the prediction horizon grows, or which features are most predictive.

**Example**: To replicate a full experiment for one stock, you might do:  
```bash
# Step 1: Construct features for stock 002352.XSHE (for example)
# (Make sure PreTrain.py is set to process the desired stock(s))
python PreTrain.py  

# Step 2: Train and evaluate models on calendar-time splits for that stock
python "Train copy calendar.py"  

# Step 3: Train and evaluate models on transaction-time or intraday segments
python "periodic train.py"  
```  
Monitor the output at each stage. The scripts might take a while to run, given the large volume of high-frequency data and the use of hyperparameter tuning. It’s recommended to start with a smaller subset of data (or fewer trials in optuna) to ensure everything works, then scale up to the full dataset for final results.


## Methodology  
This project implements the methodology from the referenced paper, focusing on how to predict short-term price movements with high-frequency data and determining under what conditions such predictions are feasible. The key components of the methodology include:

- **Feature Engineering (Predictors)**: A rich set of predictors is constructed from the limit order book data. The code computes 13 base features over multiple short time windows. These features, drawn from market microstructure theory, include:  
  - *Breadth*: the difference in the number of trades on the bid vs. ask side (buy vs. sell pressure) over a window.  
  - *Immediacy*: a measure related to the urgency of trades (for instance, how quickly trades execute – possibly the inverse of waiting time or an indicator of very recent trades).  
  - *Volume (All/Avg/Max)*: total traded volume, average trade size, and maximum trade size in the window. These capture the level of activity and presence of large trades.  
  - *Lambda*: the intensity of trades or the hazard rate (commonly, λ in durations modeling indicates how quickly the next trade arrives).  
  - *LOB Imbalance*: order book imbalance – the difference between bid and ask depths (volume available) in the limit order book, indicating supply/demand skew.  
  - *Transaction Imbalance*: imbalance in the number of buyer-initiated versus seller-initiated trades.  
  - *Past Return*: the price return over the window (how much the price moved recently).  
  - *Turnover*: the fraction of shares (or dollar value) traded relative to the stock’s float or some benchmark, in that window.  
  - *AutoCov*: autocovariance of recent returns or sign of trades (captures momentum or mean-reversion tendencies in microprice movements).  
  - *Quoted Spread*: the bid-ask spread (difference between best ask and best bid prices) – a measure of liquidity and trading cost.  
  - *Effective Spread*: the effective spread paid in trades (related to trade price vs. mid-price, indicating price impact).  

  These 13 features are computed **across multiple time scales** – the paper uses exponentially increasing window sizes (e.g., 0–0.1 seconds, 0.1–0.2s, 0.2–0.4s, ... up to ~25.6s) for calendar time, and similarly in **transaction time** (windows measured in number of trades). In the code, a list of time intervals `time_cal` defines the 9 windows for calendar time from 0.1s to 25.6s. Each base feature is calculated for each window, resulting in up to 13 × 9 = 117 features (though it appears 108 are used, possibly excluding some combinations or due to multicollinearity filtering). The feature names are concatenated with the window index (e.g., `Breadth0, Breadth1, ..., Immediacy0, ... EffectiveSpread8`). This extensive feature set captures the state of the market in the moments leading up to each prediction point.  

- **Prediction Targets**: There are two main prediction targets studied – (1) the **short-term return** (price change) following a given moment, and (2) the **duration until the next event** (such as the next trade or quote change). In practice, the code’s `yset1` likely corresponds to the next *k*-second return or the indicator of price up/down movement, and `yset2` corresponds to the time until the next trade or next quote update. The paper treated return prediction as both a regression (predicting the magnitude of return) and a classification (predicting the direction of the price move) problem, and duration as a regression (or classification of short vs. long wait). This code primarily sets them up as regression problems (using regressors like RandomForestRegressor, Lasso, etc., which output a numeric prediction), but the direction can be inferred from the sign of the return prediction as well.  

- **Machine Learning Models**: A variety of machine learning models are applied to predict the outcomes from the features:  
  - **Lasso Regression** and **Ridge Regression**: linear models with L1 and L2 regularization, respectively. These are useful for high-dimensional data to perform feature selection (Lasso) or shrinkage (Ridge). The Lasso in particular can zero out uninformative features, helping identify which predictors matter most .  
  - **Random Forests**: ensemble of decision trees, good for capturing nonlinear relationships and interactions. The code uses `RandomForestRegressor` from scikit-learn. Random forests also provide feature importance measures, which the researchers can use to gauge the importance of each microstructure feature.  
  - **Gradient Boosted Trees**: the code includes LightGBM (`LGBMRegressor`) and XGBoost (`XGBRegressor`) models, which are more advanced tree-based ensembles that often yield high predictive performance on structured data. These are also used to validate the robustness of findings across different model types.  
  - **Neural Network**: a Multi-Layer Perceptron regressor (`MLPRegressor` from scikit-learn) is used as a simple neural network model . This can capture complex nonlinear patterns, though with this many features, training an MLP might be challenging without careful tuning.  
  - **Others**: The mention of `RFscore`, `Lassoscore`, etc., in the code indicates an automated hyperparameter tuning routine (via Optuna) for Random Forest, Lasso, Ridge, LightGBM, XGBoost, and a neural net. Each model’s hyperparameters (like number of trees, regularization strength, learning rate, etc.) are optimized on a validation set to ensure comparisons are fair and each model is as predictive as possible.
