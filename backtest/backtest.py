import pandas as pd
from statsmodels.tsa.stattools import adfuller

def adf_test(series_dict_or_series):
    """
    Perform ADF test. Accepts a Series or a dictionary like:
    {
        'TOS': {
            'log_returns': pd.Series,
            'simple_returns': pd.Series
        },
        ...
    }
    """
    if isinstance(series_dict_or_series, dict):
        results = {}
        for symbol, data in series_dict_or_series.items():
            if isinstance(data, dict) and 'log_returns' in data:
                print(f"\nAnalyzing {symbol}...")
                results[symbol] = adf_test(data['log_returns'])
            else:
                raise ValueError(f"Data for {symbol} must contain 'log_returns'.")
        return results
    
    # Handle single Series input
    series = series_dict_or_series.dropna()
    result = adfuller(series)
    stats = {
        'test_statistic': result[0],
        'p_value': result[1],
        'is_stationary': result[1] < 0.05
    }
    
    print("\nAugmented Dickey-Fuller Test Results:")
    print(f"Test Statistic: {stats['test_statistic']:.4f}")
    print(f"p-value: {stats['p_value']:.4f}")
    print(f"Stationary: {'Yes' if stats['is_stationary'] else 'No'}")
    
    return stats
 


def describe_statistics(series_dict_or_series):
    """
    Calculate statistical measures (mean, std, skewness, kurtosis)
    for a single time series or a dictionary of time series.

    Parameters:
    -----------
    series_dict_or_series : pd.Series or dict
        Either a single time series (log returns), or a dictionary like:
        {
            'TOS': {'log_returns': pd.Series, ...},
            ...
        }

    Returns:
    --------
    dict: statistical measures (but no raw print)
    """
    if isinstance(series_dict_or_series, dict):
        results = {}
        for symbol, data in series_dict_or_series.items():
            if isinstance(data, dict) and 'log_returns' in data:
                print(f"\n📊 Analyzing {symbol}:")
                results[symbol] = describe_statistics(data['log_returns'])
            else:
                raise ValueError(f"Data for {symbol} must contain 'log_returns'.")
        return results

    # Handle Series input
    series = series_dict_or_series.dropna()
    stats = {
        'mean': float(series.mean()),
        'std': float(series.std()),
        'skew': float(series.skew()),
        'kurtosis': float(series.kurtosis())
    }

    # Print formatted results
    print(f"Mean: {stats['mean']:.6f}")
    print(f"Standard Deviation: {stats['std']:.6f}")
    print(f"Skewness: {stats['skew']:.4f}")
    print(f"Kurtosis: {stats['kurtosis']:.4f}")

    return stats