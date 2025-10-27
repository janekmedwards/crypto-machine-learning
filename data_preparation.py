"""
Data Preparation Pipeline for Crypto Arbitrage Forecasting

This module prepares real-time cryptocurrency data for machine learning models,
including feature engineering, technical indicator calculation, and feature selection.
"""

import os
import sqlite3
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import StandardScaler
from ta.momentum import RSIIndicator
from ta.trend import SMAIndicator, EMAIndicator
from ta.volatility import BollingerBands

warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)

# Constants
PREDICTION_HORIZON = 60  # minutes ahead for prediction
RESAMPLE_FREQ = '1T'  # 1-minute intervals
TIME_FRAMES = [10, 30, 60, 120]  # minutes for technical indicators
CORRELATION_THRESHOLD = 0.8


def load_data(db_path="realtime_crypto_data.db"):
    """Load and preprocess raw crypto data from SQLite database."""
    conn = sqlite3.connect(db_path)
    df = pd.read_sql("SELECT * FROM realtime_crypto_data", conn)
    conn.close()
    
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)
    df.sort_index(inplace=True)
    
    return df


def select_features(df):
    """Select relevant columns for analysis."""
    return df[['btc_usd_ask', 'btc_usd_ask_volume', 'btc_usd_bid_volume', 'btc_zar_bid',
               'btc_zar_ask_volume', 'btc_zar_bid_volume', 'usd_zar_rate', 'forward_arbitrage']]


def resample_data(df):
    """Resample data to fixed intervals and forward fill missing values."""
    initial_cov = df.shape[0]
    df_resampled = df.resample(RESAMPLE_FREQ).ffill()
    coverage = (initial_cov / df_resampled.shape[0]) * 100
    print(f"Coverage Before Forward Fill: {coverage:.3f}%")
    return df_resampled


def create_target_variable(df):
    """Create target variable by shifting forward_arbitrage."""
    df['target_1hr'] = df['forward_arbitrage'].shift(-PREDICTION_HORIZON)
    df.dropna(inplace=True)
    return df


def create_depth_features(df):
    """Calculate market depth features from volume ratios."""
    df['btc_usd_ask_depth'] = df['btc_usd_ask_volume'] / (df['btc_usd_ask_volume'] + df['btc_usd_bid_volume'])
    df['btc_usd_bid_depth'] = df['btc_usd_bid_volume'] / (df['btc_usd_ask_volume'] + df['btc_usd_bid_volume'])
    df['btc_zar_ask_depth'] = df['btc_zar_ask_volume'] / (df['btc_zar_ask_volume'] + df['btc_zar_bid_volume'])
    df['btc_zar_bid_depth'] = df['btc_zar_bid_volume'] / (df['btc_zar_ask_volume'] + df['btc_zar_bid_volume'])
    return df


def select_final_features(df):
    """Select final feature set for analysis."""
    return df[['btc_usd_ask', 'btc_usd_ask_depth', 'btc_usd_bid_depth', 'btc_zar_bid',
               'btc_zar_ask_depth', 'btc_zar_bid_depth', 'usd_zar_rate',
               'forward_arbitrage', 'target_1hr']]


def generate_visualizations(df):
    """Generate time series plots, correlation heatmap, and histograms."""
    os.makedirs('visualizations', exist_ok=True)
    plt.style.use('ggplot')
    
    # Time series plots
    for column in df.columns:
        plt.figure(figsize=(15, 6))
        plt.plot(df.index, df[column])
        plt.title(f'{column} Over Time')
        plt.xlabel('Time')
        plt.ylabel(column)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'visualizations/{column}_time_series.png')
        plt.close()
    
    # Correlation heatmap
    plt.figure(figsize=(12, 8))
    correlation_matrix = df.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title('Correlation Matrix')
    plt.tight_layout()
    plt.savefig('visualizations/correlation_matrix.png')
    plt.close()
    
    # Histograms
    for column in df.columns:
        plt.figure(figsize=(10, 6))
        plt.hist(df[column], bins=50)
        plt.title(f'Histogram of {column}')
        plt.xlabel(column)
        plt.ylabel('Frequency')
        plt.tight_layout()
        plt.savefig(f'visualizations/{column}_histogram.png')
        plt.close()


def winsorize_data(df, lower_percentile=0.01, upper_percentile=0.99):
    """Handle outliers by clipping values to specified percentiles."""
    df_winsorized = df.copy()
    for col in df.columns:
        lower = df[col].quantile(lower_percentile)
        upper = df[col].quantile(upper_percentile)
        df_winsorized[col] = df[col].clip(lower=lower, upper=upper)
    return df_winsorized


def create_technical_features(df, price_cols=None, volume_cols=None):
    """
    Create technical indicators for multiple timeframes.
    
    Parameters:
    - df: DataFrame containing price and volume data
    - price_cols: List of price columns to calculate indicators for
    - volume_cols: List of volume columns to calculate indicators for
    
    Returns:
    - DataFrame with added technical indicators
    """
    if price_cols is None:
        price_cols = ['btc_usd_ask', 'btc_zar_bid', 'forward_arbitrage', 'usd_zar_rate']
    if volume_cols is None:
        volume_cols = ['btc_usd_ask_depth', 'btc_zar_bid_depth']
    
    df_features = df.copy()
    
    # Add time-based features
    df_features['hour'] = df_features.index.hour
    df_features['day_of_week'] = df_features.index.dayofweek
    
    # Calculate technical indicators for each price column
    for col in price_cols:
        for tf in TIME_FRAMES:
            # Moving Averages
            df_features[f'{col}_sma_{tf}'] = SMAIndicator(close=df[col], window=tf).sma_indicator()
            df_features[f'{col}_ema_{tf}'] = EMAIndicator(close=df[col], window=tf).ema_indicator()
            
            # Bollinger Bands
            bb = BollingerBands(close=df[col], window=tf, window_dev=2)
            df_features[f'{col}_bb_upper_{tf}'] = bb.bollinger_hband()
            df_features[f'{col}_bb_lower_{tf}'] = bb.bollinger_lband()
            df_features[f'{col}_bb_middle_{tf}'] = bb.bollinger_mavg()
            df_features[f'{col}_bb_width_{tf}'] = (
                (df_features[f'{col}_bb_upper_{tf}'] - df_features[f'{col}_bb_lower_{tf}']) /
                df_features[f'{col}_bb_middle_{tf}']
            )
            
            # RSI
            df_features[f'{col}_rsi_{tf}'] = RSIIndicator(close=df[col], window=tf).rsi()
            
            # Momentum and volatility
            df_features[f'{col}_momentum_{tf}'] = df[col].pct_change(tf)
            df_features[f'{col}_volatility_{tf}'] = df[col].rolling(window=tf).std()
            df_features[f'{col}_range_{tf}'] = (
                df[col].rolling(window=tf).max() - df[col].rolling(window=tf).min()
            )
            df_features[f'{col}_roc_{tf}'] = df[col].pct_change(tf) * 100
    
    # Create cross-market features
    for tf in TIME_FRAMES:
        df_features[f'price_spread_{tf}'] = (
            df_features[f'btc_usd_ask_sma_{tf}'] - df_features[f'btc_zar_bid_sma_{tf}']
        )
        df_features[f'relative_strength_{tf}'] = (
            df_features[f'btc_usd_ask_rsi_{tf}'] - df_features[f'btc_zar_bid_rsi_{tf}']
        )
        df_features[f'volatility_ratio_{tf}'] = (
            df_features[f'btc_usd_ask_volatility_{tf}'] /
            df_features[f'btc_zar_bid_volatility_{tf}']
        )
    
    # Create volume-based features
    for vol_col in volume_cols:
        for tf in TIME_FRAMES:
            df_features[f'{vol_col}_sma_{tf}'] = df[vol_col].rolling(window=tf).mean()
            df_features[f'{vol_col}_momentum_{tf}'] = df[vol_col].pct_change(tf)
            df_features[f'{vol_col}_volatility_{tf}'] = df[vol_col].rolling(window=tf).std()
    
    df_features.dropna(inplace=True)
    return df_features


def analyze_feature_importance(df, target_col='target_1hr', top_n=50, correlation_threshold=None):
    """
    Analyze feature importance using mutual information and create visualizations.
    
    Parameters:
    - df: DataFrame containing features and target
    - target_col: Name of the target column
    - top_n: Number of top features to select
    - correlation_threshold: Threshold for correlation filtering
    
    Returns:
    - DataFrame containing selected features and their importance scores
    """
    if correlation_threshold is None:
        correlation_threshold = CORRELATION_THRESHOLD
    
    os.makedirs('feature_importance', exist_ok=True)
    
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Handle infinite values
    X = X.replace([np.inf, -np.inf], np.nan).ffill()
    
    # Scale and calculate mutual information
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    mi_scores = mutual_info_regression(X_scaled, y)
    
    # Create importance DataFrame
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': mi_scores
    }).sort_values('Importance', ascending=False)
    
    # Select top features accounting for correlations
    initial_top_features = feature_importance.head(top_n * 2)
    top_feature_cols = initial_top_features['Feature'].tolist()
    correlation_matrix = df[top_feature_cols].corr().abs()
    
    def find_correlated_groups(correlation_matrix, threshold):
        """Find groups of correlated features."""
        groups = []
        remaining_features = set(correlation_matrix.columns)
        
        while remaining_features:
            current_feature = remaining_features.pop()
            current_group = {current_feature}
            
            for feature in list(remaining_features):
                if correlation_matrix.loc[current_feature, feature] > threshold:
                    current_group.add(feature)
                    remaining_features.remove(feature)
            
            groups.append(current_group)
        
        return groups
    
    # Select best feature from each correlated group
    correlated_groups = find_correlated_groups(correlation_matrix, correlation_threshold)
    selected_features = []
    
    for group in correlated_groups:
        group_importance = feature_importance[feature_importance['Feature'].isin(group)]
        best_feature = group_importance.iloc[0]['Feature']
        selected_features.append(best_feature)
        
        if len(selected_features) >= top_n:
            break
    
    final_features = feature_importance[feature_importance['Feature'].isin(selected_features)]
    
    # Visualize results
    plt.figure(figsize=(15, 8))
    sns.barplot(x='Importance', y='Feature', data=final_features)
    plt.title(f'Top {len(selected_features)} Most Important Features (Correlation Filtered)')
    plt.xlabel('Mutual Information Score')
    plt.tight_layout()
    plt.savefig('feature_importance/top_features.png')
    plt.close()
    
    correlation_matrix = df[selected_features + [target_col]].corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
    plt.title('Correlation Matrix of Selected Features')
    plt.tight_layout()
    plt.savefig('feature_importance/correlation_heatmap.png')
    plt.close()
    
    print(f"\nSelected {len(selected_features)} Features (after correlation filtering):")
    print(final_features)
    print("\nFeature importance analysis complete. Results saved in 'feature_importance' directory.")
    
    return final_features


def analyze_xgboost_features(df, target_col='target_1hr', top_n=150, correlation_threshold=None):
    """
    Analyze features specifically for XGBoost model using gain-based importance.
    
    Parameters:
    - df: DataFrame containing features and target
    - target_col: Name of the target column
    - top_n: Number of top features to select
    - correlation_threshold: Threshold for correlation filtering
    
    Returns:
    - DataFrame containing selected features and their importance scores
    """
    import xgboost as xgb
    from sklearn.model_selection import train_test_split
    
    if correlation_threshold is None:
        correlation_threshold = CORRELATION_THRESHOLD
    
    os.makedirs('xgboost_features', exist_ok=True)
    
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Handle missing and infinite values
    X = X.replace([np.inf, -np.inf], np.nan).fillna(X.mean())
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
    
    # Split and train XGBoost model
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    
    model = xgb.XGBRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    # Get feature importance scores
    importance_scores = model.get_booster().get_score(importance_type='gain')
    feature_importance = pd.DataFrame({
        'Feature': list(importance_scores.keys()),
        'Importance': list(importance_scores.values())
    }).sort_values('Importance', ascending=False)
    
    # Select top features accounting for correlations
    initial_top_features = feature_importance.head(top_n * 2)
    top_feature_cols = initial_top_features['Feature'].tolist()
    correlation_matrix = X[top_feature_cols].corr().abs()
    
    selected_features = []
    for feature in top_feature_cols:
        if not selected_features:
            selected_features.append(feature)
        else:
            correlations = correlation_matrix.loc[feature, selected_features]
            if not any(correlations > correlation_threshold):
                selected_features.append(feature)
        
        if len(selected_features) >= top_n:
            break
    
    final_features = feature_importance[feature_importance['Feature'].isin(selected_features)]
    
    # Visualize results
    plt.figure(figsize=(15, 8))
    sns.barplot(x='Importance', y='Feature', data=final_features)
    plt.title(f'Top {len(selected_features)} Most Important Features for XGBoost')
    plt.xlabel('Gain Importance Score')
    plt.tight_layout()
    plt.savefig('xgboost_features/feature_importance.png')
    plt.close()
    
    # Correlation heatmap
    corr_data = pd.concat([X[selected_features], y], axis=1)
    correlation_matrix = corr_data.corr()
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
    plt.title('Correlation Matrix of Selected Features')
    plt.tight_layout()
    plt.savefig('xgboost_features/correlation_heatmap.png')
    plt.close()
    
    # Feature interaction plots
    for i, feat1 in enumerate(selected_features[:3]):
        for feat2 in selected_features[i+1:4]:
            plt.figure(figsize=(10, 6))
            plt.scatter(X[feat1], X[feat2], c=y, cmap='viridis', alpha=0.5)
            plt.colorbar(label=target_col)
            plt.xlabel(feat1)
            plt.ylabel(feat2)
            plt.title(f'Feature Interaction: {feat1} vs {feat2}')
            plt.tight_layout()
            plt.savefig(f'xgboost_features/interaction_{feat1}_vs_{feat2}.png')
            plt.close()
    
    print(f"\nSelected {len(selected_features)} Features for XGBoost:")
    print(final_features)
    print("\nFeature importance analysis complete. Results saved in 'xgboost_features' directory.")
    
    return final_features


# Main execution
if __name__ == "__main__":
    # Load and preprocess data
    df = load_data()
    df = select_features(df)
    df.describe().to_csv('data_description.csv')
    df = resample_data(df)
    df = create_target_variable(df)
    
    # Create depth features and generate visualizations
    df = create_depth_features(df)
    df_f1 = select_final_features(df)
    generate_visualizations(df_f1)
    
    # Winsorize data and create technical features
    df_winsorized = winsorize_data(df_f1)
    df_features = create_technical_features(df_winsorized)
    
    # Analyze features for both LSTM and XGBoost models
    lstm_features_1 = analyze_feature_importance(df_features, top_n=30)
    lstm_features_2 = analyze_feature_importance(df_features, top_n=50)
    xgboost_features = analyze_xgboost_features(df_features, top_n=150)
    
    # Create datasets using selected features
    lstm_feature_cols_1 = lstm_features_1['Feature'].tolist() + ['target_1hr']
    lstm_feature_cols_2 = lstm_features_2['Feature'].tolist() + ['target_1hr']
    xgboost_feature_cols = xgboost_features['Feature'].tolist() + ['target_1hr']
    
    # Create final datasets for model training
    df_lstm_1 = df_features[lstm_feature_cols_1]
    df_lstm_2 = df_features[lstm_feature_cols_2]
    df_xgboost = df_features[xgboost_feature_cols]
    df_1v = df_winsorized[['forward_arbitrage', 'target_1hr']]