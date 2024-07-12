# Time Series Predictability 

This repository contains a script to classify time series based on their next data point predictability using various statistical metrics and features. The goal is to identify and rank time series according to their predictability and classify them into different groups.

## Table of Contents
- [Introduction](#introduction)
- [Time Series Analysis](#time-series-analysis)
- [Statistical Metrics and Features](#statistical-metrics-and-features)
- [Structure](#structure)
- [Usage](#usage)
- [Configuration](#configuration)
- [Implementation Details](#implementation-details)
- [Output](#output)
- [Contributing](#contributing)
- [License](#license)

## Introduction

Time series analysis is a crucial aspect of data science, used to analyze sequential data points collected over time. This repository demonstrates how to classify time series based on their predictability using various statistical methods.

## Time Series Analysis

A time series is a series of data points indexed in time order, typically with uniform intervals. Time series analysis involves understanding the underlying patterns such as trends, seasonality, and noise to make predictions.

### Components of Time Series
1. **Trend**: The long-term movement in the data.
2. **Seasonality**: The repeating short-term cycle in the data.
3. **Noise**: The random variation in the data.

## Statistical Metrics and Features

The script calculates several statistical metrics and features to evaluate and rank the predictability of each time series. Here's an explanation of each:

1. **SMAPE (Symmetric Mean Absolute Percentage Error)**:
   - A measure of accuracy of predictive models.
   - Formula: ![SMAPE](https://wikimedia.org/api/rest_v1/media/math/render/svg/51f203c983bde9c771fbe89f62d93739c4a3795f)
   - Symmetric and prevents issues with values near zero.

2. **MAPE (Mean Absolute Percentage Error)**:
   - Measures the accuracy as a percentage of the error.
   - Formula: ![MAPE](https://wikimedia.org/api/rest_v1/media/math/render/svg/f0a53ed8eb9c0f5111f9b76d89d1d3f25f5677d3)

3. **AIC (Akaike Information Criterion)**:
   - Measures the quality of a model relative to other models.
   - Lower AIC indicates a better fit.

4. **Degree of Differencing**:
   - Indicates the number of times the data needs to be differenced to achieve stationarity.

5. **Autoregressive (AR) Terms**:
   - The number of lag observations included in the model.

6. **Variance**:
   - Measures the spread of the data points.

7. **Seasonality**:
   - Presence of repeating patterns at regular intervals.

8. **Holiday Effect**:
   - The impact of holidays on the data.

9. **Spikes and Dips**:
   - Presence of sudden increases or decreases in the data.

## Structure

- `main.py`: The main script to execute the classification process.
- `metrics.py`: Module containing functions to calculate metrics and handle preprocessing.
- `config.py`: Configuration file for defining weights and other parameters.
- `requirements.txt`: List of dependencies.
- `outputs/`: Folder where the classification results are saved as a JSON file.

## Usage

1. **Install the required dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
2. **Run the main script**:
    ```bash
    python main.py
    ```
3. **Results**:
    The results will be saved into outputs/results.json file, containing the names of the time series in each group.

## Configuration
```python
# config.py

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
```

## Implementation Details

### Data Preparation
The script uses synthetic data for demonstration purposes. Replace the sample data with your actual time series data.

### Metrics Calculation
The calculate_metrics function in metrics.py computes the various metrics for each time series. The metrics are normalized and weighted according to the configuration.

### Ranking and Classification
Time series are ranked based on the composite score, calculated by summing the weighted metrics. The ranked series are then classified into groups.

### Saving Results
The results are saved as a single JSON file (outputs/results.json) with keys Group_1, Group_2, and Group_3, each containing the corresponding time series.

### Output
The output is a JSON file (outputs/results.json) structured as follows:
```json
{
    "Group_1": ["TS_1", "TS_2", "TS_3", ...],
    "Group_2": ["TS_4", "TS_5", "TS_6", ...],
    "Group_3": ["TS_7", "TS_8", "TS_9", ...]
}
````

## Contributing
Contributions are welcome! Please fork the repository and submit a pull request.

