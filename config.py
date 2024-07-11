# Define weights for each metric
weights = {
    'smape': 0.2,
    'mape': 0.2,
    'aic': 0.1,
    'degree_of_differencing': 0.1,
    'ar_terms': 0.1,
    'variance': 0.1,
    'seasonality': 0.1,
    'holiday_effect': 0.05,
    'spikes_and_dips': 0.05
}

# Define other parameters as needed
train_size_ratio = 0.8