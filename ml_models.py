"""
Machine Learning Models for Crypto Arbitrage Forecasting

This module implements LSTM and XGBoost models with hyperparameter optimization
and time series cross-validation for predicting cryptocurrency arbitrage opportunities.
"""

import json
import os

import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb

import matplotlib.pyplot as plt
from scikeras.wrappers import KerasRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import RobustScaler, StandardScaler
from skopt.space import Categorical, Integer, Real
from skopt import gp_minimize
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

from data_preparation import df_1v, df_lstm_1, df_lstm_2, df_xgboost


def simple_lstm(data, target_col='target_1hr', sequence_length=90, n_splits=5, output_dir='lstm_results'):
    """
    Implements a simple LSTM model with time series cross-validation.

    Parameters:
    - data: DataFrame containing features and target
    - target_col: Name of the target column
    - sequence_length: Length of input sequences for LSTM
    - n_splits: Number of cross-validation splits
    - output_dir: Directory to save results

    Returns:
    - Trained model, scalers, and performance metrics
    """
    os.makedirs(output_dir, exist_ok=True)
    
    X = data.drop(columns=[target_col])
    y = data[target_col]
    
    tscv = TimeSeriesSplit(n_splits=n_splits)
    rmse_scores, mae_scores, r2_scores = [], [], []
    
    def create_sequences(X, y, seq_length):
        """Create sequences for LSTM input."""
        Xs, ys = [], []
        for i in range(len(X) - seq_length):
            Xs.append(X[i:(i + seq_length)])
            ys.append(y[i + seq_length])
        return np.array(Xs), np.array(ys)
    
    for fold, (train_idx, test_idx) in enumerate(tscv.split(X), 1):
        print(f"\nTraining Fold {fold}/{n_splits}")
        
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        scaler_X = RobustScaler()
        scaler_y = RobustScaler()
        
        X_train_scaled = scaler_X.fit_transform(X_train)
        X_test_scaled = scaler_X.transform(X_test)
        y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1))
        y_test_scaled = scaler_y.transform(y_test.values.reshape(-1, 1))
        
        X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train_scaled, sequence_length)
        X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test_scaled, sequence_length)
        
        model = Sequential([
            LSTM(64, activation='tanh', return_sequences=True, 
                 input_shape=(sequence_length, X_train.shape[1]),
                 recurrent_dropout=0.1),
            LSTM(32, activation='tanh', recurrent_dropout=0.1),
            Dense(16, activation='relu'),
            Dense(1)
        ])
        
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.001)
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        
        model.compile(optimizer='adam', loss='huber')
        
        history = model.fit(
            X_train_seq, y_train_seq,
            epochs=100,
            batch_size=45,
            validation_split=0.1,
            callbacks=[reduce_lr, early_stopping],
            verbose=1
        )
        
        y_pred_scaled = model.predict(X_test_seq)
        y_pred = scaler_y.inverse_transform(y_pred_scaled)
        y_test_actual = scaler_y.inverse_transform(y_test_seq)
        
        rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred))
        mae = mean_absolute_error(y_test_actual, y_pred)
        r2 = r2_score(y_test_actual, y_pred)
        
        rmse_scores.append(rmse)
        mae_scores.append(mae)
        r2_scores.append(r2)
        
        plt.figure(figsize=(15, 6))
        plt.plot(y_test_actual, label='Actual')
        plt.plot(y_pred, label='Predicted')
        plt.title(f'Fold {fold} - Actual vs Predicted Values')
        plt.xlabel('Time')
        plt.ylabel(target_col)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'{output_dir}/fold_{fold}_predictions.png')
        plt.close()
        
        plt.figure(figsize=(15, 6))
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title(f'Fold {fold} - Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'{output_dir}/fold_{fold}_training_history.png')
        plt.close()
    
    avg_rmse = np.mean(rmse_scores)
    avg_mae = np.mean(mae_scores)
    avg_r2 = np.mean(r2_scores)
    
    print("\nCross-Validation Results:")
    print(f"Average RMSE: {avg_rmse:.4f}")
    print(f"Average MAE: {avg_mae:.4f}")
    print(f"Average R2 Score: {avg_r2:.4f}")
    
    with open(f'{output_dir}/cv_metrics.txt', 'w') as f:
        f.write(f"Average RMSE: {avg_rmse:.4f}\n")
        f.write(f"Average MAE: {avg_mae:.4f}\n")
        f.write(f"Average R2 Score: {avg_r2:.4f}\n")
        f.write("\nMetrics per fold:\n")
        for fold in range(n_splits):
            f.write(f"\nFold {fold + 1}:\n")
            f.write(f"RMSE: {rmse_scores[fold]:.4f}\n")
            f.write(f"MAE: {mae_scores[fold]:.4f}\n")
            f.write(f"R2 Score: {r2_scores[fold]:.4f}\n")
    
    return model, scaler_X, scaler_y, (avg_rmse, avg_mae, avg_r2)


def xgboost_model(data, target_col='target_1hr', n_splits=5, output_dir='xgboost_results'):
    """
    Implements an XGBoost model with time series cross-validation.

    Parameters:
    - data: DataFrame containing features and target
    - target_col: Name of the target column
    - n_splits: Number of cross-validation splits
    - output_dir: Directory to save results

    Returns:
    - Trained model, scaler, and performance metrics
    """
    os.makedirs(output_dir, exist_ok=True)
    
    X = data.drop(columns=[target_col])
    y = data[target_col]
    
    X = X.replace([np.inf, -np.inf], np.nan).fillna(X.mean())
    
    tscv = TimeSeriesSplit(n_splits=n_splits)
    rmse_scores, mae_scores, r2_scores = [], [], []
    feature_importance_scores = []
    
    for fold, (train_idx, test_idx) in enumerate(tscv.split(X), 1):
        print(f"\nTraining Fold {fold}/{n_splits}")
        
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=X.columns)
        
        model = xgb.XGBRegressor(
            n_estimators=50,
            learning_rate=0.3,
            max_depth=3,
            min_child_weight=1,
            subsample=1.0,
            colsample_bytree=0.6,
            gamma=0.0,
            reg_alpha=1.0,
            reg_lambda=1.0,
            random_state=42
        )
        
        model.fit(
            X_train_scaled, y_train,
            eval_set=[(X_test_scaled, y_test)],
            verbose=False
        )
        
        y_pred = model.predict(X_test_scaled)
        
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        rmse_scores.append(rmse)
        mae_scores.append(mae)
        r2_scores.append(r2)
        
        importance = model.get_booster().get_score(importance_type='gain')
        feature_importance_scores.append(importance)
        
        plt.figure(figsize=(15, 6))
        plt.plot(y_test.values, label='Actual')
        plt.plot(y_pred, label='Predicted')
        plt.title(f'Fold {fold} - Actual vs Predicted Values')
        plt.xlabel('Time')
        plt.ylabel(target_col)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'{output_dir}/fold_{fold}_predictions.png')
        plt.close()
    
    avg_rmse = np.mean(rmse_scores)
    avg_mae = np.mean(mae_scores)
    avg_r2 = np.mean(r2_scores)
    
    print("\nCross-Validation Results:")
    print(f"Average RMSE: {avg_rmse:.4f}")
    print(f"Average MAE: {avg_mae:.4f}")
    print(f"Average R2 Score: {avg_r2:.4f}")
    
    avg_importance = {}
    for feature in X.columns:
        scores = [fold_importance.get(feature, 0) for fold_importance in feature_importance_scores]
        avg_importance[feature] = np.mean(scores)
    
    importance_df = pd.DataFrame({
        'Feature': list(avg_importance.keys()),
        'Importance': list(avg_importance.values())
    }).sort_values('Importance', ascending=False)
    
    plt.figure(figsize=(15, 8))
    sns.barplot(x='Importance', y='Feature', data=importance_df)
    plt.title('Average Feature Importance Across Folds')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/average_feature_importance.png')
    plt.close()
    
    with open(f'{output_dir}/cv_metrics.txt', 'w') as f:
        f.write(f"Average RMSE: {avg_rmse:.4f}\n")
        f.write(f"Average MAE: {avg_mae:.4f}\n")
        f.write(f"Average R2 Score: {avg_r2:.4f}\n")
        f.write("\nMetrics per fold:\n")
        for fold in range(n_splits):
            f.write(f"\nFold {fold + 1}:\n")
            f.write(f"RMSE: {rmse_scores[fold]:.4f}\n")
            f.write(f"MAE: {mae_scores[fold]:.4f}\n")
            f.write(f"R2 Score: {r2_scores[fold]:.4f}\n")
    
    return model, scaler, (avg_rmse, avg_mae, avg_r2)


def optimize_lstm_hyperparameters(data, target_col='target_1hr', output_dir='lstm_optimization'):
    """
    Optimize LSTM hyperparameters using Bayesian optimization.

    Parameters:
    - data: DataFrame containing features and target
    - target_col: Name of the target column
    - output_dir: Directory to save optimization results

    Returns:
    - Dictionary containing best hyperparameters
    """
    os.makedirs(output_dir, exist_ok=True)
    
    X = data.drop(columns=[target_col])
    y = data[target_col]
    
    for col in X.columns:
        lower = X[col].quantile(0.01)
        upper = X[col].quantile(0.99)
        X[col] = X[col].clip(lower=lower, upper=upper)
    
    scaler_X = RobustScaler()
    scaler_y = RobustScaler()
    
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1))
    
    def create_sequences(X, y, seq_length):
        """Create sequences for LSTM input."""
        Xs, ys = [], []
        for i in range(len(X) - seq_length):
            Xs.append(X[i:(i + seq_length)])
            ys.append(y[i + seq_length])
        return np.array(Xs), np.array(ys)
    
    def create_model(units_layer1, units_layer2, dropout_rate, learning_rate, sequence_length):
        """Create LSTM model with specified architecture."""
        model = Sequential([
            LSTM(units_layer1, activation='tanh', return_sequences=True, 
                 input_shape=(sequence_length, X.shape[1]),
                 recurrent_dropout=dropout_rate),
            Dropout(dropout_rate),
            LSTM(units_layer2, activation='tanh', 
                 recurrent_dropout=dropout_rate),
            Dropout(dropout_rate),
            Dense(16, activation='relu'),
            Dense(1)
        ])
        
        optimizer = Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss='huber')
        return model
    
    sequence_lengths = [60, 90]
    sequence_data = {}
    
    for seq_len in sequence_lengths:
        X_seq, y_seq = create_sequences(X_scaled, y_scaled, seq_len)
        sequence_data[seq_len] = (X_seq, y_seq)
    
    tscv = TimeSeriesSplit(n_splits=3)
    results = []
    
    for seq_len in sequence_lengths:
        print(f"\nOptimizing for sequence length: {seq_len}")
        X_seq, y_seq = sequence_data[seq_len]
        
        cv_splits = list(tscv.split(X_seq))
        
        def objective(params):
            """Objective function for Bayesian optimization."""
            units1 = params['units_layer1']
            units2 = params['units_layer2']
            dropout = params['dropout_rate']
            lr = params['learning_rate']
            batch_size = params['batch_size']
            
            cv_scores = []
            
            for train_idx, val_idx in cv_splits:
                model = create_model(
                    units_layer1=units1,
                    units_layer2=units2,
                    dropout_rate=dropout,
                    learning_rate=lr,
                    sequence_length=seq_len
                )
                
                early_stopping = EarlyStopping(
                    monitor='val_loss',
                    patience=5,
                    restore_best_weights=True
                )
                
                history = model.fit(
                    X_seq[train_idx], y_seq[train_idx],
                    validation_data=(X_seq[val_idx], y_seq[val_idx]),
                    epochs=50,
                    batch_size=batch_size,
                    callbacks=[early_stopping],
                    verbose=0
                )
                
                val_loss = min(history.history['val_loss'])
                cv_scores.append(val_loss)
            
            avg_score = np.mean(cv_scores)
            
            results.append({
                'sequence_length': seq_len,
                'units_layer1': units1,
                'units_layer2': units2,
                'dropout_rate': dropout,
                'learning_rate': lr,
                'batch_size': batch_size,
                'score': avg_score
            })
            
            print(f"Parameters: {params}")
            print(f"Average CV score: {avg_score:.4f}")
            
            return avg_score
        
        param_space = [
            (32, 128),
            (16, 64),
            (0.1, 0.3),
            (0.0001, 0.001),
            [32, 64]
        ]
        
        result = gp_minimize(
            func=lambda x: objective({
                'units_layer1': int(x[0]),
                'units_layer2': int(x[1]),
                'dropout_rate': x[2],
                'learning_rate': x[3],
                'batch_size': x[4]
            }),
            dimensions=param_space,
            n_calls=15,
            random_state=42,
            verbose=True
        )
        
        best_params = {
            'units_layer1': int(result.x[0]),
            'units_layer2': int(result.x[1]),
            'dropout_rate': result.x[2],
            'learning_rate': result.x[3],
            'batch_size': result.x[4],
            'sequence_length': seq_len
        }
        best_score = result.fun
        
        print(f"\nBest parameters for sequence length {seq_len}:")
        print(best_params)
        print(f"Best score: {best_score:.4f}")
    
    results_df = pd.DataFrame(results)
    best_result = results_df.loc[results_df['score'].idxmin()]
    
    results_df.to_csv(f'{output_dir}/all_results.csv', index=False)
    
    plt.figure(figsize=(15, 8))
    for seq_len in sequence_lengths:
        seq_results = results_df[results_df['sequence_length'] == seq_len]
        plt.scatter(seq_results['score'], [seq_len] * len(seq_results), 
                   label=f'Sequence Length {seq_len}')
    
    plt.xlabel('Validation Loss')
    plt.ylabel('Sequence Length')
    plt.title('Optimization Results by Sequence Length')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{output_dir}/optimization_results.png')
    plt.close()
    
    best_config = best_result.to_dict()
    with open(f'{output_dir}/best_configuration.json', 'w') as f:
        json.dump(best_config, f, indent=4)
    
    print("\nBest Configuration:")
    print(f"Sequence Length: {best_config['sequence_length']}")
    print(f"Units Layer 1: {best_config['units_layer1']}")
    print(f"Units Layer 2: {best_config['units_layer2']}")
    print(f"Dropout Rate: {best_config['dropout_rate']}")
    print(f"Learning Rate: {best_config['learning_rate']}")
    print(f"Batch Size: {best_config['batch_size']}")
    print(f"Score: {best_config['score']:.4f}")
    
    return best_config


def lstm_optimized(data, target_col='target_1hr', output_dir='lstm_optimized_results'):
    """
    LSTM model using optimized hyperparameters from Bayesian optimization.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    sequence_length = int(90)
    units_layer1 = int(104)
    units_layer2 = int(64)
    dropout_rate = 0.23360052491561492
    learning_rate = 0.00014128439258985422
    batch_size = int(45)
    
    X = data.drop(columns=[target_col])
    y = data[target_col]
    
    for col in X.columns:
        lower = X[col].quantile(0.01)
        upper = X[col].quantile(0.99)
        X[col] = X[col].clip(lower=lower, upper=upper)
    
    tscv = TimeSeriesSplit(n_splits=5)
    rmse_scores, mae_scores, r2_scores = [], [], []
    
    def create_sequences(X, y, seq_length):
        Xs, ys = [], []
        for i in range(len(X) - seq_length):
            Xs.append(X[i:(i + seq_length)])
            ys.append(y[i + seq_length])
        return np.array(Xs), np.array(ys)
    
    for fold, (train_idx, test_idx) in enumerate(tscv.split(X), 1):
        print(f"\nTraining Fold {fold}/5")
        
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        scaler_X = RobustScaler()
        scaler_y = RobustScaler()
        
        X_train_scaled = scaler_X.fit_transform(X_train)
        X_test_scaled = scaler_X.transform(X_test)
        y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1))
        y_test_scaled = scaler_y.transform(y_test.values.reshape(-1, 1))
        
        X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train_scaled, sequence_length)
        X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test_scaled, sequence_length)
        
        model = Sequential([
            LSTM(units_layer1, activation='tanh', return_sequences=True, 
                 input_shape=(sequence_length, X_train.shape[1]),
                 recurrent_dropout=dropout_rate),
            Dropout(dropout_rate),
            LSTM(units_layer2, activation='tanh', 
                 recurrent_dropout=dropout_rate),
            Dropout(dropout_rate),
            Dense(16, activation='relu'),
            Dense(1)
        ])
        
        optimizer = Adam(learning_rate=learning_rate)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001)
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        
        model.compile(optimizer=optimizer, loss='huber')
        
        history = model.fit(
            X_train_seq, y_train_seq,
            epochs=100,
            batch_size=batch_size,
            validation_split=0.1,
            callbacks=[reduce_lr, early_stopping],
            verbose=1
        )
        
        y_pred_scaled = model.predict(X_test_seq)
        y_pred = scaler_y.inverse_transform(y_pred_scaled)
        y_test_actual = scaler_y.inverse_transform(y_test_seq)
        
        rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred))
        mae = mean_absolute_error(y_test_actual, y_pred)
        r2 = r2_score(y_test_actual, y_pred)
        
        rmse_scores.append(rmse)
        mae_scores.append(mae)
        r2_scores.append(r2)
        
        plt.figure(figsize=(15, 6))
        plt.plot(y_test_actual, label='Actual')
        plt.plot(y_pred, label='Predicted')
        plt.title(f'Fold {fold} - Actual vs Predicted Values')
        plt.xlabel('Time')
        plt.ylabel(target_col)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'{output_dir}/fold_{fold}_predictions.png')
        plt.close()
        
        plt.figure(figsize=(15, 6))
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title(f'Fold {fold} - Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'{output_dir}/fold_{fold}_training_history.png')
        plt.close()
    
    avg_rmse = np.mean(rmse_scores)
    avg_mae = np.mean(mae_scores)
    avg_r2 = np.mean(r2_scores)
    
    print("\nCross-Validation Results:")
    print(f"Average RMSE: {avg_rmse:.4f}")
    print(f"Average MAE: {avg_mae:.4f}")
    print(f"Average R2 Score: {avg_r2:.4f}")
    
    with open(f'{output_dir}/cv_metrics.txt', 'w') as f:
        f.write(f"Average RMSE: {avg_rmse:.4f}\n")
        f.write(f"Average MAE: {avg_mae:.4f}\n")
        f.write(f"Average R2 Score: {avg_r2:.4f}\n")
        f.write("\nMetrics per fold:\n")
        for fold in range(5):
            f.write(f"\nFold {fold + 1}:\n")
            f.write(f"RMSE: {rmse_scores[fold]:.4f}\n")
            f.write(f"MAE: {mae_scores[fold]:.4f}\n")
            f.write(f"R2 Score: {r2_scores[fold]:.4f}\n")
    
    return model, scaler_X, scaler_y, (avg_rmse, avg_mae, avg_r2)


def optimize_xgboost_hyperparameters(data, target_col='target_1hr', output_dir='xgboost_optimization'):
    """
    Optimize XGBoost hyperparameters using Bayesian optimization.

    Parameters:
    - data: DataFrame containing features and target
    - target_col: Name of the target column
    - output_dir: Directory to save optimization results

    Returns:
    - Dictionary containing best hyperparameters
    """
    os.makedirs(output_dir, exist_ok=True)
    
    X = data.drop(columns=[target_col])
    y = data[target_col]
    
    X = X.replace([np.inf, -np.inf], np.nan).fillna(X.mean())
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
    
    tscv = TimeSeriesSplit(n_splits=3)
    results = []
    
    param_space = [
        Integer(50, 300, name='n_estimators'),
        Real(0.01, 0.3, prior='log-uniform', name='learning_rate'),
        Integer(3, 10, name='max_depth'),
        Integer(1, 10, name='min_child_weight'),
        Real(0.6, 1.0, name='subsample'),
        Real(0.6, 1.0, name='colsample_bytree'),
        Real(0.0, 0.5, name='gamma'),
        Real(0.0, 1.0, name='reg_alpha'),
        Real(0.0, 1.0, name='reg_lambda')
    ]
    
    def objective(params):
        n_estimators, learning_rate, max_depth, min_child_weight, subsample, colsample_bytree, gamma, reg_alpha, reg_lambda = params
        
        cv_scores = []
        
        for train_idx, val_idx in tscv.split(X_scaled):
            X_train, X_val = X_scaled.iloc[train_idx], X_scaled.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            model = xgb.XGBRegressor(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                max_depth=max_depth,
                min_child_weight=min_child_weight,
                subsample=subsample,
                colsample_bytree=colsample_bytree,
                gamma=gamma,
                reg_alpha=reg_alpha,
                reg_lambda=reg_lambda,
                random_state=42
            )
            
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False
            )
            
            val_score = model.score(X_val, y_val)
            cv_scores.append(val_score)
        
        avg_score = np.mean(cv_scores)
        
        results.append({
            'n_estimators': n_estimators,
            'learning_rate': learning_rate,
            'max_depth': max_depth,
            'min_child_weight': min_child_weight,
            'subsample': subsample,
            'colsample_bytree': colsample_bytree,
            'gamma': gamma,
            'reg_alpha': reg_alpha,
            'reg_lambda': reg_lambda,
            'score': avg_score
        })
        
        print(f"Parameters: {params}")
        print(f"Average CV score: {avg_score:.4f}")
        
        return avg_score
    
    result = gp_minimize(
        func=objective,
        dimensions=param_space,
        n_calls=20,
        random_state=42,
        verbose=True
    )
    
    best_params = {
        'n_estimators': int(result.x[0]),
        'learning_rate': result.x[1],
        'max_depth': int(result.x[2]),
        'min_child_weight': int(result.x[3]),
        'subsample': result.x[4],
        'colsample_bytree': result.x[5],
        'gamma': result.x[6],
        'reg_alpha': result.x[7],
        'reg_lambda': result.x[8]
    }
    best_score = result.fun
    
    results_df = pd.DataFrame(results)
    results_df.to_csv(f'{output_dir}/all_results.csv', index=False)
    
    plt.figure(figsize=(15, 8))
    plt.plot(results_df['score'])
    plt.title('Optimization Progress')
    plt.xlabel('Iteration')
    plt.ylabel('Validation Score')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/optimization_progress.png')
    plt.close()
    
    param_importance = {}
    for param in best_params.keys():
        if param != 'score':
            correlation = np.corrcoef(results_df[param], results_df['score'])[0, 1]
            param_importance[param] = abs(correlation)
    
    plt.figure(figsize=(12, 6))
    plt.bar(param_importance.keys(), param_importance.values())
    plt.title('Parameter Importance')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/parameter_importance.png')
    plt.close()
    
    with open(f'{output_dir}/best_configuration.json', 'w') as f:
        json.dump(best_params, f, indent=4)
    
    print("\nBest Configuration:")
    for param, value in best_params.items():
        print(f"{param}: {value}")
    print(f"Best score: {best_score:.4f}")
    
    return best_params


# Main execution
if __name__ == "__main__":
    # Run simple LSTM models
    model_tf_simple_lstm_1, scaler_X_tf_simple_lstm_1, scaler_y_tf_simple_lstm_1, metrics_tf_simple_lstm_1 = simple_lstm(df_1v)
    model_tf_simple_lstm_2, scaler_X_tf_simple_lstm_2, scaler_y_tf_simple_lstm_2, metrics_tf_simple_lstm_2 = simple_lstm(df_lstm_1)
    model_tf_simple_lstm_3, scaler_X_tf_simple_lstm_3, scaler_y_tf_simple_lstm_3, metrics_tf_simple_lstm_3 = simple_lstm(df_lstm_2)
    
    # Run LSTM hyperparameter optimization
    best_config_lstm = optimize_lstm_hyperparameters(df_1v)
    model_tf_lstm, scaler_X_tf_lstm, scaler_y_tf_lstm, metrics_tf_lstm = lstm_optimized(df_1v)
    
    # Run XGBoost hyperparameter optimization
    best_config_xgboost = optimize_xgboost_hyperparameters(df_xgboost)
    model_tf_xgboost, scaler_tf_xgboost, metrics_tf_xgboost = xgboost_model(df_xgboost)