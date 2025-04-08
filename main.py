#!/usr/bin/env python
"""
Time Series Classifier Script

This script classifies time series based on predictability metrics.
"""
import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from time_series_predictability.metrics import calculate_metrics, normalize_metrics
from time_series_predictability.config import weights
from time_series_predictability.utils import load_data, setup_logging


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Classify time series by predictability")
    parser.add_argument(
        "--input", 
        type=str, 
        help="Path to input CSV file with time series data",
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default="outputs/results.json", 
        help="Output JSON file path (default: outputs/results.json)",
    )
    parser.add_argument(
        "--train-ratio", 
        type=float, 
        default=0.8, 
        help="Ratio of data to use for training (default: 0.8)",
    )
    parser.add_argument(
        "--groups", 
        type=int, 
        default=3, 
        help="Number of groups to classify into (default: 3)",
    )
    parser.add_argument(
        "--log-level", 
        type=str, 
        default="INFO", 
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level",
    )
    return parser.parse_args()


def classify_time_series(
    data: pd.DataFrame, 
    train_size_ratio: float, 
    num_groups: int, 
    weights: Dict[str, float]
) -> Dict[str, List[str]]:
    """
    Classify time series based on predictability metrics.
    
    Args:
        data: DataFrame with time series as columns
        train_size_ratio: Ratio of data to use for training
        num_groups: Number of groups to classify into
        weights: Dictionary of metric weights
        
    Returns:
        Dictionary with group names as keys and lists of time series as values
    """
    logging.info(f"Calculating metrics for {data.shape[1]} time series")
    
    # Calculate metrics for each time series
    metrics = {}
    for column in data.columns:
        try:
            metrics[column] = calculate_metrics(data[column], train_size_ratio)
        except Exception as e:
            logging.warning(f"Error calculating metrics for {column}: {e}")
            continue
            
    # Create a DataFrame from the metrics
    metrics_df = pd.DataFrame(metrics).T
    
    # Normalize metrics
    normalized_metrics = normalize_metrics(metrics_df)
    
    # Calculate weighted scores
    for metric, weight in weights.items():
        if metric in normalized_metrics.columns:
            normalized_metrics[metric] *= weight
        else:
            logging.warning(f"Metric '{metric}' specified in weights but not found in calculated metrics")
    
    # Combine normalized and weighted metrics into a composite score
    normalized_metrics['composite_score'] = normalized_metrics.sum(axis=1)
    
    # Rank time series based on composite score
    sorted_scores = normalized_metrics['composite_score'].sort_values()
    
    # Divide into groups
    group_size = len(sorted_scores) // num_groups
    groups = {}
    
    for i in range(num_groups):
        group_name = f'Group_{i+1}'
        if i == num_groups - 1:
            # Last group takes all remaining time series
            groups[group_name] = sorted_scores[i * group_size:].index.tolist()
        else:
            groups[group_name] = sorted_scores[i * group_size:(i + 1) * group_size].index.tolist()
    
    return groups


def main() -> None:
    """Main function to run the time series classification."""
    args = parse_arguments()
    setup_logging(args.log_level)
    
    # Ensure the output directory exists
    output_path = Path(args.output)
    os.makedirs(output_path.parent, exist_ok=True)
    
    try:
        # Load data
        if args.input:
            data = load_data(args.input)
            logging.info(f"Loaded data from {args.input} with shape {data.shape}")
        else:
            # Generate sample data if no input file is provided
            import numpy as np
            logging.warning("No input file provided, generating random data")
            data = pd.DataFrame(
                np.random.randn(36, 100), 
                columns=[f'TS_{i}' for i in range(100)]
            )
        
        # Classify time series
        groups = classify_time_series(
            data=data,
            train_size_ratio=args.train_ratio,
            num_groups=args.groups,
            weights=weights
        )
        
        # Save results
        with open(output_path, 'w') as f:
            json.dump(groups, f, indent=4)
        
        logging.info(f"Results saved to {output_path}")
        
        # Output the groups
        for group, series in groups.items():
            logging.info(f"{group}: {len(series)} time series")
            
    except Exception as e:
        logging.error(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()