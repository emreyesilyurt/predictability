import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_absolute_percentage_error

def calculate_metrics(time_series, train_size_ratio=0.8):
    train_size = int(len(time_series) * train_size_ratio)
    train, test = time_series[0:train_size], time_series[train_size:]
    model = ARIMA(train, order=(5, 1, 0))
    model_fit = model.fit()
    predictions = model_fit.forecast(steps=len(test))
    
    # Calculate errors
    smape = 100/len(test) * np.sum(2 * np.abs(predictions - test) / (np.abs(test) + np.abs(predictions)))
    mape = mean_absolute_percentage_error(test, predictions)
    
    # Calculate AIC
    aic = model_fit.aic
    
    # Degree of Differencing
    result = adfuller(train)
    p_value = result[1]
    degree_of_differencing = 0 if p_value < 0.05 else 1
    
    # Autoregressive terms (order)
    ar_terms = 5  # as specified in ARIMA(5, 1, 0)
    
    # Variance
    variance = np.var(train)
    
    # Seasonality (simplified)
    seasonality = np.std(train[1:] - train[:-1])
    
    # Placeholder for Holiday Effect, Spikes and Dips
    holiday_effect = 0  # assuming no data on holidays effect
    spikes_and_dips = np.sum(np.abs(train - np.mean(train)) > 2 * np.std(train))
    
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

def normalize_metrics(metrics_df):
    return (metrics_df - metrics_df.mean()) / metrics_df.std()
