import numpy as np
import pandas as pd

# Monte carlo simulation
def monte_carlo_compare(data_dict, days=30, simulations=100000):
    """
    Run Monte Carlo simulation for multiple stock symbols and compare results

    Parameters:
    -----------
    data_dict : dict
        Dictionary containing DataFrames from get_stock_data
    days : int
        Number of days to simulate (default 252 days = 1 year)
    simulations : int
        Number of simulations per stock symbol

    Returns:
    --------
    pd.DataFrame
        Comparison results for stocks: return, risk, VaR 5%
    """
    results = []

    for symbol, df in data_dict.items():
        df = df.sort_values("time").copy()
        df['log_return'] = np.log(df['close'] / df['close'].shift(1))
        returns = df['log_return'].dropna()

        mu_r = returns.mean()
        sigma_r = returns.std()
        start_price = df['close'].iloc[-1]

        # Simulate
        simulated_returns = np.random.normal(mu_r, sigma_r, (simulations, days))
        price_paths = start_price * np.exp(np.cumsum(simulated_returns, axis=1))
        final_prices = price_paths[:, -1]
        future_returns = ((final_prices - start_price) / start_price)*100

        results.append({
            'Symbol': symbol,
            'Expected Return': np.mean(future_returns),
            'Risk (Std Dev)': np.std(future_returns),
            'VaR 5%': np.percentile(future_returns, 5),
            'Current Price': start_price
        })

    return pd.DataFrame(results).sort_values("Expected Return", ascending=False)

# Example usage:
# results = monte_carlo_simulation(df, mu_r, sigma_r)  # Commented out example usage