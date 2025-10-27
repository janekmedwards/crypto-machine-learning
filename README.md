# Machine Learning Models for Crypto Arbitrage Forecasting

This project implements advanced machine learning models for predicting cryptocurrency arbitrage opportunities using LSTM and XGBoost algorithms.

## Overview

This repository contains a comprehensive machine learning pipeline for forecasting cryptocurrency arbitrage opportunities. The project includes:

1. **Data Preparation**: Feature engineering, technical indicators, and data preprocessing
2. **Model Training**: LSTM and XGBoost models with cross-validation
3. **Hyperparameter Optimization**: Bayesian optimization for model tuning
4. **Evaluation**: Comprehensive performance metrics and visualizations

## Project Structure

```
├── data_preparation.py        # Data preprocessing and feature engineering
├── ml_models.py              # ML model implementations
├── requirements.txt           # Python dependencies
├── README.md                 # This file
├── lstm_results/             # LSTM model outputs
├── xgboost_results/           # XGBoost model outputs
├── feature_importance/        # Feature analysis outputs
├── xgboost_features/          # XGBoost feature analysis
└── visualizations/            # Data visualizations
```

## Features

### Data Preparation (`data_preparation.py`)

- **Data Loading**: Loads real-time cryptocurrency data from SQLite database
- **Technical Indicators**: Generates multiple technical indicators (RSI, SMA, EMA, Bollinger Bands)
- **Feature Engineering**: Creates market depth, momentum, volatility, and cross-market features
- **Feature Selection**: Uses mutual information and XGBoost gain-based importance
- **Data Preprocessing**: Winsorization, outlier handling, and scaling

### Machine Learning Models (`ml_models.py`)

#### LSTM Models
- **Simple LSTM**: Basic LSTM architecture with time series cross-validation
- **Optimized LSTM**: Hyperparameter-tuned LSTM using Bayesian optimization
- Architecture includes:
  - Recurrent dropout for regularization
  - Early stopping and learning rate reduction
  - Multiple LSTM layers with dense output

#### XGBoost Model
- Gradient boosting with time series cross-validation
- Feature importance analysis
- Hyperparameter optimization via Bayesian methods

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### 1. Data Preparation

Run the data preparation pipeline:

```bash
python data_preparation.py
```

This will:
- Load data from `realtime_crypto_data.db`
- Create technical features and indicators
- Perform feature importance analysis
- Generate visualizations
- Create datasets for LSTM and XGBoost models

### 2. Model Training

Run the machine learning models:

```bash
python ml_models.py
```

This will:
- Train multiple LSTM models with different configurations
- Train XGBoost models with cross-validation
- Perform hyperparameter optimization
- Generate performance metrics and visualizations

### 3. Running Individual Models

You can also run specific model functions from within Python:

```python
from data_preparation import df_lstm_1, df_xgboost
from ml_models import simple_lstm, lstm_optimized, xgboost_model

# Run simple LSTM
model, scaler_X, scaler_y, metrics = simple_lstm(df_lstm_1, output_dir='my_results')

# Run optimized LSTM
model, scaler_X, scaler_y, metrics = lstm_optimized(df_lstm_1, output_dir='my_results')

# Run XGBoost
model, scaler, metrics = xgboost_model(df_xgboost, output_dir='my_results')
```

## Configuration

### Key Parameters

#### Data Preparation
- `PREDICTION_HORIZON`: 60 minutes (future prediction window)
- `RESAMPLE_FREQ`: '1T' (1-minute intervals)
- `TIME_FRAMES`: [10, 30, 60, 120] minutes for technical indicators
- `CORRELATION_THRESHOLD`: 0.8 (for feature selection)

#### Model Training
- `sequence_length`: 90 (for LSTM input sequences)
- `n_splits`: 5 (cross-validation folds)
- `batch_size`: 45
- `epochs`: 100 (with early stopping)

### Output Directories

Results are organized into several directories:
- `lstm_results/`: LSTM model outputs and metrics
- `xgboost_results/`: XGBoost model outputs and metrics
- `feature_importance/`: Feature analysis for LSTM models
- `xgboost_features/`: Feature analysis for XGBoost
- `visualizations/`: Data visualizations
- `lstm_optimization/`: Hyperparameter optimization results
- `xgboost_optimization/`: XGBoost optimization results

## Model Performance

### Metrics

Each model evaluation includes:
- **RMSE**: Root Mean Squared Error
- **MAE**: Mean Absolute Error
- **R² Score**: Coefficient of determination

### Output Files

For each model, the following files are generated:
- `cv_metrics.txt`: Cross-validation metrics summary
- `fold_{n}_predictions.png`: Prediction vs actual plots per fold
- `fold_{n}_training_history.png`: Training history plots
- `average_feature_importance.png`: Feature importance visualization (XGBoost)

## Key Functions

### Data Preparation Functions

- `load_data(db_path)`: Load data from SQLite database
- `select_features(df)`: Select relevant columns
- `resample_data(df)`: Resample to fixed intervals
- `create_target_variable(df)`: Create prediction target
- `create_depth_features(df)`: Calculate market depth features
- `create_technical_features(df)`: Generate technical indicators
- `analyze_feature_importance(df)`: Analyze features for LSTM
- `analyze_xgboost_features(df)`: Analyze features for XGBoost
- `winsorize_data(df)`: Handle outliers

### Model Functions

- `simple_lstm(data)`: Train basic LSTM model
- `lstm_optimized(data)`: Train optimized LSTM with hyperparameters
- `xgboost_model(data)`: Train XGBoost model
- `optimize_lstm_hyperparameters(data)`: Bayesian optimization for LSTM
- `optimize_xgboost_hyperparameters(data)`: Bayesian optimization for XGBoost

## Data Requirements

### Input Data Format

The SQLite database (`realtime_crypto_data.db`) should contain the following columns:

- `btc_usd_ask`: BTC/USD ask price
- `btc_usd_ask_volume`: BTC/USD ask volume
- `btc_usd_bid_volume`: BTC/USD bid volume
- `btc_zar_bid`: BTC/ZAR bid price
- `btc_zar_ask_volume`: BTC/ZAR ask volume
- `btc_zar_bid_volume`: BTC/ZAR bid volume
- `usd_zar_rate`: USD/ZAR exchange rate
- `forward_arbitrage`: Arbitrage spread
- `timestamp`: DateTime index

## Customization

### Adapting for Different Data

1. **Update database schema**: Modify `select_features()` to match your database columns
2. **Adjust technical indicators**: Modify time frames in `TIME_FRAMES` constant
3. **Change prediction horizon**: Update `PREDICTION_HORIZON` constant
4. **Modify model architecture**: Edit model creation functions in `ml_models.py`

### Changing Model Configuration

To modify LSTM architecture:
```python
# In ml_models.py, edit the model creation:
model = Sequential([
    LSTM(64, ...),  # Change number of units
    Dropout(0.2),   # Adjust dropout rate
    LSTM(32, ...),
    ...
])
```

To modify XGBoost parameters:
```python
# In ml_models.py, edit the model:
model = xgb.XGBRegressor(
    n_estimators=100,    # Change number of trees
    learning_rate=0.1,    # Adjust learning rate
    max_depth=5,         # Modify tree depth
    ...
)
```

## Troubleshooting

### Common Issues

1. **Memory errors with large datasets**: 
   - Reduce batch size
   - Use smaller sequence lengths
   - Enable subsampling

2. **Poor model performance**:
   - Increase feature selection count
   - Try different hyperparameters
   - Check data quality and preprocessing

3. **Import errors**:
   - Ensure all dependencies are installed
   - Check Python version compatibility
   - Verify database file exists

### Performance Tips

- Use GPU acceleration for LSTM training (configure TensorFlow)
- Reduce cross-validation folds for faster iteration
- Enable early stopping to prevent overfitting
- Use feature selection to reduce dimensionality

## Author

Janek Masojada Edwards

## Acknowledgments

- TensorFlow/Keras for LSTM implementation
- XGBoost for gradient boosting
- Scikit-learn for preprocessing and evaluation
- Technical Analysis Library (ta) for indicators
