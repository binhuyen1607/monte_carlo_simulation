import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.tsa.stattools import adfuller
import seaborn as sns  # Added for plot_risk_return function

# Visualization
def analyze_stock_returns(df):
    """
    Analyze and visualize stock returns before and after log transformation
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing stock data with 'close' column
    """
    # Sort data by time
    df = df.sort_values(by="time")
    
    # Calculate simple returns and log returns
    df['simple_return'] = df['close'].pct_change()
    df['log_return'] = np.log(df['close'] / df['close'].shift(1))
    simple_returns = df['simple_return'].dropna()
    log_returns = df['log_return'].dropna()
    
    # Perform stationarity tests
    simple_test = adfuller(simple_returns)
    simple_pvalue = simple_test[1]
    
    log_test = adfuller(log_returns)
    log_pvalue = log_test[1]
    
    # Create figure with 2x2 subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot simple returns
    ax1.plot(simple_returns)
    ax1.set_title(f'Simple Returns (p-value: {simple_pvalue:.4f})')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Simple Returns')
    ax1.grid(True)
    
    # Plot simple returns histogram with normal curve
    ax2.hist(simple_returns, bins=50, density=True, alpha=0.7)
    x = np.linspace(simple_returns.min(), simple_returns.max(), 100)
    mu = simple_returns.mean()
    sigma = simple_returns.std()
    ax2.plot(x, stats.norm.pdf(x, mu, sigma), 'r-', lw=2)
    ax2.set_title('Simple Returns Distribution')
    ax2.set_xlabel('Simple Returns')
    ax2.set_ylabel('Frequency')
    
    # Plot log returns
    ax3.plot(log_returns)
    ax3.set_title(f'Log Returns (p-value: {log_pvalue:.4f})')
    ax3.set_xlabel('Time')
    ax3.set_ylabel('Log Returns')
    ax3.grid(True)
    
    # Plot log returns histogram with normal curve
    ax4.hist(log_returns, bins=50, density=True, alpha=0.7)
    x = np.linspace(log_returns.min(), log_returns.max(), 100)
    mu = log_returns.mean()
    sigma = log_returns.std()
    ax4.plot(x, stats.norm.pdf(x, mu, sigma), 'r-', lw=2)
    ax4.set_title('Log Returns Distribution')
    ax4.set_xlabel('Log Returns')
    ax4.set_ylabel('Frequency')
    
    plt.tight_layout()
    plt.show()
    
    # Print stationarity test results
    print(f"Simple returns ADF test p-value: {simple_pvalue:.4f}")
    print(f"Log returns ADF test p-value: {log_pvalue:.4f}")
    print("\nInterpretation:")
    print("p-value < 0.05 indicates stationarity")
    print("p-value >= 0.05 indicates non-stationarity")
    
    return log_returns

def analyze_stock_returns(data_dict):
    """
    Analyze and visualize stock returns before and after log transformation for multiple symbols
    
    Parameters:
    -----------
    data_dict : dict
        Dictionary containing DataFrames for each symbol with 'close' column
    """
    results = {}
    
    for symbol, df in data_dict.items():
        print(f"\nAnalyzing {symbol}...")
        
        # Sort data by time
        df = df.sort_values(by="time")
        
        # Calculate simple returns and log returns
        df['simple_return'] = df['close'].pct_change()
        df['log_return'] = np.log(df['close'] / df['close'].shift(1))
        simple_returns = df['simple_return'].dropna()
        log_returns = df['log_return'].dropna()
        
        # Perform stationarity tests
        simple_test = adfuller(simple_returns)
        simple_pvalue = simple_test[1]
        
        log_test = adfuller(log_returns)
        log_pvalue = log_test[1]
        
        # Create figure with 2x2 subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Returns Analysis for {symbol}', fontsize=16)
        
        # Plot simple returns
        ax1.plot(simple_returns)
        ax1.set_title(f'Simple Returns (p-value: {simple_pvalue:.4f})')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Simple Returns')
        ax1.grid(True)
        
        # Plot simple returns histogram with normal curve
        ax2.hist(simple_returns, bins=50, density=True, alpha=0.7)
        x = np.linspace(simple_returns.min(), simple_returns.max(), 100)
        mu = simple_returns.mean()
        sigma = simple_returns.std()
        ax2.plot(x, stats.norm.pdf(x, mu, sigma), 'r-', lw=2)
        ax2.set_title('Simple Returns Distribution')
        ax2.set_xlabel('Simple Returns')
        ax2.set_ylabel('Frequency')
        
        # Plot log returns
        ax3.plot(log_returns)
        ax3.set_title(f'Log Returns (p-value: {log_pvalue:.4f})')
        ax3.set_xlabel('Time')
        ax3.set_ylabel('Log Returns')
        ax3.grid(True)
        
        # Plot log returns histogram with normal curve
        ax4.hist(log_returns, bins=50, density=True, alpha=0.7)
        x = np.linspace(log_returns.min(), log_returns.max(), 100)
        mu = log_returns.mean()
        sigma = log_returns.std()
        ax4.plot(x, stats.norm.pdf(x, mu, sigma), 'r-', lw=2)
        ax4.set_title('Log Returns Distribution')
        ax4.set_xlabel('Log Returns')
        ax4.set_ylabel('Frequency')
        
        plt.tight_layout()
        plt.show()
        
        # Print stationarity test results
        print(f"Simple returns ADF test p-value: {simple_pvalue:.4f}")
        print(f"Log returns ADF test p-value: {log_pvalue:.4f}")
        print("\nInterpretation:")
        print("p-value < 0.05 indicates stationarity")
        print("p-value >= 0.05 indicates non-stationarity")
        
        # Store results
        results[symbol] = {
            'log_returns': log_returns,
            'simple_returns': simple_returns,
            'simple_pvalue': simple_pvalue,
            'log_pvalue': log_pvalue
        }
    
    return results

def visualize_price_and_returns(df_dict, symbol="Stock"):
    """
    Plot price and log return charts for all symbols in the dictionary
    
    Parameters:
    -----------
    df_dict : dict
        Dictionary containing DataFrames for each symbol
    symbol : str
        Symbol to highlight in the plot
    """
    # Create figure with subplots for each symbol
    n_symbols = len(df_dict)
    fig, axs = plt.subplots(n_symbols, 2, figsize=(15, 5*n_symbols), sharex=True)
    
    for idx, (sym, df) in enumerate(df_dict.items()):
        df = df.sort_values("time").copy()
        df['log_return'] = np.log(df['close'] / df['close'].shift(1))
        
        # Price chart
        axs[idx, 0].plot(df["time"], df["close"], color='blue')
        axs[idx, 0].set_title(f'Stock Price {sym}', fontsize=14)
        axs[idx, 0].set_ylabel('Closing Price')
        axs[idx, 0].grid(True)
        
        # Log return chart
        axs[idx, 1].plot(df["time"], df["log_return"], color='green', alpha=0.7)
        axs[idx, 1].axhline(0, color='gray', lw=1, linestyle='--')
        axs[idx, 1].set_title(f'Log Returns {sym}', fontsize=14)
        axs[idx, 1].set_ylabel('Log Return')
        axs[idx, 1].set_xlabel('Time')
        axs[idx, 1].grid(True)
    
    plt.tight_layout()
    plt.show()
    
def plot_risk_return(df):
    """
    Plot scatter chart: Expected Return vs Risk
    """
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='Risk (Std Dev)', y='Expected Return', hue='Symbol', s=120)
    plt.title('Risk vs Return Comparison (Monte Carlo 1 year)')
    plt.xlabel('Risk (Standard Deviation)')
    plt.ylabel('Expected Return')
    plt.grid(True)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

# plot_risk_return(compare_df)    
def plot_individual_stocks(data_dict, cols=3):
    """
    Plot price movements for each stock, one subplot per symbol
    Parameters:
    -----------
    data_dict : dict
        Dictionary containing stock DataFrames
    cols : int
        Number of columns in subplot layout
    """
    import math

    total = len(data_dict)
    rows = math.ceil(total / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), sharex=False)
    axes = axes.flatten()  # for easier handling

    for idx, (symbol, df) in enumerate(data_dict.items()):
        ax = axes[idx]
        df_sorted = df.sort_values("time")
        ax.plot(df_sorted['time'], df_sorted['close'], label=symbol)
        ax.set_title(symbol)
        ax.set_xlabel("Time")
        ax.set_ylabel("Closing Price")
        ax.grid(True)

    # Hide empty subplots (if number of symbols is not divisible)
    for i in range(len(data_dict), len(axes)):
        fig.delaxes(axes[i])

    plt.suptitle("Price Movements of Individual Stocks", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # leave space for title
    plt.show()
#plot_individual_stocks(df, cols=3)  # Or cols=3 if you want a different layout