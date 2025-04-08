"""
Time Series Metrics Module

This module provides functions to calculate various predictability metrics for time series data.
"""
import logging
from typing import Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_absolute_percentage_error


def split_time_series(
    time_series: pd.Series, 
    train_size_ratio: float = 0.8
) -> Tuple[pd.Series, pd.Series]:
    """
    Split a time series into training and testing sets.
    
    Args:
        time_series: Input time series as pandas Series
        train_size_ratio: Proportion of data to use for training
        
    Returns:
        Tuple of (train, test) series
    """
    if not 0 < train_size_ratio < 1:
        raise ValueError("train_size_ratio must be between 0 and 1")
        
    train_size = int(len(time_series) * train_size_ratio)
    if train_size < 5:
        raise ValueError(f"Training set too small ({train_size} points)")
        
    return time_series[0:train_size], time_series[train_size:]


def fit_arima_model(
    train_data: pd.Series, 
    order: Tuple[int, int, int] = (5, 1, 0)
) -> Tuple[ARIMA, np.ndarray]:
    """
    Fit an ARIMA model to training data and return the model and predictions.
    
    Args:
        train_data: Training time series data
        order: ARIMA model order (p, d, q)
        
    Returns:
        Tuple of (fitted_model, predictions)
    """
    model = ARIMA(train_data, order=order)
    try:
        model_fit = model.fit()
        return model_fit, None
    except Exception as e:
        logging.warning(f"Error fitting ARIMA model: {e}")
        # Try with different order if the initial fit fails
        try:
            fallback_order = (1, 1, 0)
            logging.info(f"Trying fallback ARIMA order {fallback_order}")
            model = ARIMA(train_data, order=fallback_order)
            model_fit = model.fit()
            return model_fit, None
        except Exception as e2:
            logging.error(f"Error fitting fallback ARIMA model: {e2}")
            raise ValueError(f"Could not fit ARIMA model: {e2}")


def forecast_time_series(
    model_fit, 
    steps: int
) -> np.ndarray:
    """
    Generate forecasts from a fitted model.
    
    Args:
        model_fit: Fitted ARIMA model
        steps: Number of steps to forecast
        
    Returns:
        Array of forecasted values
    """
    try:
        predictions = model_fit.forecast(steps=steps)
        return predictions
    except Exception as e:
        logging.error(f"Error forecasting: {e}")
        raise ValueError(f"Could not generate forecasts: {e}")


def calculate_smape(
    actual: np.ndarray, 
    predicted: np.ndarray
) -> float:
    """
    Calculate Symmetric Mean Absolute Percentage Error.
    
    Args:
        actual: Actual values
        predicted: Predicted values
        
    Returns:
        SMAPE value (lower is better)
    """
    # Handle division by zero
    if len(actual) == 0 or len(predicted) == 0:
        return np.nan
        
    denominator = np.abs(actual) + np.abs(predicted)
    
    # Handle cases where both actual and predicted are zero
    mask = denominator != 0
    if not np.any(mask):
        return 0.0  # If all values are zero, perfect prediction
        
    # Calculate SMAPE only for non-zero denominators
    valid_actual = actual[mask]
    valid_predicted = predicted[mask]
    valid_denominator = denominator[mask]
    
    return 100/len(valid_actual) * np.sum(2 * np.abs(valid_actual - valid_predicted) / valid_denominator)


def calculate_mape(
    actual: np.ndarray, 
    predicted: np.ndarray
) -> float:
    """
    Calculate Mean Absolute Percentage Error with handling for edge cases.
    
    Args:
        actual: Actual values
        predicted: Predicted values
        
    Returns:
        MAPE value (lower is better)
    """
    # Handle cases where actual is zero
    if not np.all(actual):
        # Filter out zero values
        mask = actual != 0
        if not np.any(mask):
            return np.nan  # Can't calculate MAPE if all actual values are zero
            
        actual = actual[mask]
        predicted = predicted[mask]
        
    try:
        return mean_absolute_percentage_error(actual, predicted)
    except Exception as e:
        logging.warning(f"Error calculating MAPE: {e}")
        return np.nan


def calculate_degree_of_differencing(series: pd.Series) -> int:
    """
    Determine the degree of differencing needed for stationarity.
    
    Args:
        series: Time series data
        
    Returns:
        Degree of differencing (0, 1, or 2)
    """
    # Check original series
    result = adfuller(series)
    p_value = result[1]
    
    if p_value < 0.05:
        return 0
        
    # Check first difference
    diff1 = series.diff().dropna()
    if len(diff1) < 5:  # Need enough data points
        return 1
        
    result = adfuller(diff1)
    p_value = result[1]
    
    if p_value < 0.05:
        return 1
    
    # Check second difference
    diff2 = diff1.diff().dropna()
    if len(diff2) < 5:
        return 2
        
    result = adfuller(diff2)
    p_value = result[1]
    
    if p_value < 0.05:
        return 2
        
    return 2  # Default to 2 if still not stationary


def detect_seasonality(series: pd.Series) -> float:
    """
    Detect seasonality in a time series.
    
    Args:
        series: Time series data
        
    Returns:
        Seasonality score (higher means more seasonal)
    """
    # Simple approach: standard deviation of differences
    return np.std(series.diff().dropna())


def detect_spikes_and_dips(series: pd.Series) -> int:
    """
    Count spikes and dips in a time series.
    
    Args:
        series: Time series data
        
    Returns:
        Count of significant spikes and dips
    """
    mean = np.mean(series)
    std = np.std(series)
    
    # Count points more than 2 standard deviations from mean
    return np.sum(np.abs(series - mean) > 2 * std)


def calculate_metrics(
    time_series: pd.Series, 
    train_size_ratio: float = 0.8
) -> Dict[str, float]:
    """
    Calculate various predictability metrics for a time series.
    
    Args:
        time_series: Input time series as pandas Series
        train_size_ratio: Proportion of data to use for training
        
    Returns:
        Dictionary of calculated metrics
    """
    logging.debug(f"Calculating metrics for series of length {len(time_series)}")
    
    # Split data
    try:
        train, test = split_time_series(time_series, train_size_ratio)
    except Exception as e:
        logging.error(f"Error splitting time series: {e}")
        raise
        
    # Fit model
    try:
        model_fit, _ = fit_arima_model(train)
        aic = model_fit.aic
        ar_terms = 5  # As specified in the default order
        
        # Generate predictions
        predictions = forecast_time_series(model_fit, len(test))
    except Exception as e:
        logging.error(f"Error in model fitting or forecasting: {e}")
        # Return placeholder values if modeling fails
        return {
            'smape': np.nan,
            'mape': np.nan,
            'aic': np.nan,
            'degree_of_differencing': 1,
            'ar_terms': 5,
            'variance': np.var(train) if 'train' in locals() else np.nan,
            'seasonality': 0,
            'holiday_effect': 0,
            'spikes_and_dips': 0
        }
    
    # Calculate error metrics
    smape = calculate_smape(test.values, predictions)
    mape = calculate_mape(test.values, predictions)
    
    # Calculate other metrics
    degree_of_differencing = calculate_degree_of_differencing(train)
    variance = np.var(train)
    seasonality = detect_seasonality(train)
    spikes_and_dips = detect_spikes_and_dips(train)
    
    # Placeholder for holiday effect (would require dates)
    holiday_effect = 0
    
    return {
        'smape': smape,
        'mape': mape,
        'aic': aic,
        'degree_of_differencing': degree_of_differencing,
        'ar_terms': ar_terms,
        'variance': variance,
        'seasonality': seasonality,
        'holiday_effect': holiday_effect,
        'spikes_and_dips': spikes_and_dips
    }


def normalize_metrics(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize metrics to be comparable.
    
    Args:
        metrics_df: DataFrame of metrics with time series as index
        
    Returns:
        DataFrame of normalized metrics
    """
    # Handle NaN values
    metrics_df_clean = metrics_df.copy()
    
    # Replace NaNs with column means
    for col in metrics_df_clean.columns:
        if metrics_df_clean[col].isna().any():
            col_mean = metrics_df_clean[col].mean(skipna=True)
            metrics_df_clean[col] = metrics_df_clean[col].fillna(col_mean)
    
    # Normalize using Z-score
    return (metrics_df_clean - metrics_df_clean.mean()) / metrics_df_clean.std()