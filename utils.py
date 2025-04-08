"""
Utility functions for the time series predictability package.
"""
import logging
import sys
from pathlib import Path
from typing import Optional, Union

import pandas as pd
import numpy as np


def setup_logging(level: str = "INFO") -> None:
    """
    Set up logging with the specified level.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {level}")
        
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout
    )


def load_data(filepath: Union[str, Path]) -> pd.DataFrame:
    """
    Load time series data from a file.
    
    Args:
        filepath: Path to the data file
        
    Returns:
        DataFrame with time series data
    """
    path = Path(filepath)
    
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {filepath}")
        
    if path.suffix.lower() == '.csv':
        return pd.read_csv(filepath, index_col=0)
    elif path.suffix.lower() in ['.xls', '.xlsx']:
        return pd.read_excel(filepath, index_col=0)
    elif path.suffix.lower() == '.json':
        return pd.read_json(filepath)
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")


def save_results(results: dict, filepath: Union[str, Path]) -> None:
    """
    Save classification results to a JSON file.
    
    Args:
        results: Dictionary of classification results
        filepath: Path to save the results
    """
    import json
    path = Path(filepath)
    
    # Create directory if it doesn't exist
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, 'w') as f:
        json.dump(results, f, indent=4)
        
    logging.info(f"Results saved to {path}")


def generate_sample_data(
    n_series: int = 10, 
    n_points: int = 100,
    with_trend: bool = True,
    with_seasonality: bool = True,
    with_noise: bool = True
) -> pd.DataFrame:
    """
    Generate sample time series data for testing.
    
    Args:
        n_series: Number of time series to generate
        n_points: Number of points in each time series
        with_trend: Whether to include trend component
        with_seasonality: Whether to include seasonal component
        with_noise: Whether to include random noise
        
    Returns:
        DataFrame with generated time series as columns
    """
    data = {}
    
    for i in range(n_series):
        # Start with a base series
        series = np.zeros(n_points)
        
        # Add trend if requested
        if with_trend:
            trend_strength = np.random.uniform(0, 0.1)
            trend = np.arange(n_points) * trend_strength
            series += trend
            
        # Add seasonality if requested
        if with_seasonality:
            seasonal_strength = np.random.uniform(0.5, 2)
            period = np.random.choice([4, 6, 12, 24])
            seasonal = np.sin(np.arange(n_points) * (2 * np.pi / period)) * seasonal_strength
            series += seasonal
            
        # Add noise if requested
        if with_noise:
            noise_strength = np.random.uniform(0.1, 1)
            noise = np.random.normal(0, noise_strength, n_points)
            series += noise
            
        data[f'TS_{i}'] = series
        
    return pd.DataFrame(data)