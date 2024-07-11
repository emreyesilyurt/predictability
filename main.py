# main.py
import pandas as pd
import numpy as np
import json
import os
import warnings
from metrics import calculate_metrics, normalize_metrics
from config import weights, train_size_ratio

# Suppress specific warning
warnings.filterwarnings("ignore", message="Non-stationary starting autoregressive parameters found")

# Ensure the outputs directory exists
os.makedirs('outputs', exist_ok=True)

# Sample data preparation (replace with your actual data)
data = pd.DataFrame(np.random.randn(36, 1000), columns=[f'TS_{i}' for i in range(1000)])

# Calculate metrics for each time series
metrics = {column: calculate_metrics(data[column], train_size_ratio) for column in data.columns}

# Create a DataFrame from the metrics
metrics_df = pd.DataFrame(metrics).T

# Normalize metrics
normalized_metrics = normalize_metrics(metrics_df)

# Calculate weighted scores
for metric, weight in weights.items():
    normalized_metrics[metric] *= weight

# Combine normalized and weighted metrics into a composite score
normalized_metrics['composite_score'] = normalized_metrics.sum(axis=1)

# Rank time series based on composite score
sorted_scores = normalized_metrics['composite_score'].sort_values()

# Divide into groups
num_groups = 3
group_size = len(sorted_scores) // num_groups
groups = {}
for i in range(num_groups):
    group_name = f'Group_{i+1}'
    groups[group_name] = sorted_scores[i * group_size:(i + 1) * group_size].index.tolist()

# Save all groups to a single JSON file in the outputs folder
with open('outputs/results.json', 'w') as f:
    json.dump(groups, f, indent=4)

# Output the groups
for group, series in groups.items():
    print(f"{group}: {series}")
