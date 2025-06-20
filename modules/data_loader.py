import os
from vnstock import Quote

# get data
def get_stock_data(symbols, start_date, end_date, interval="1D"):
    """
    Fetch historical stock data for multiple symbols and save to CSV files
    
    Parameters:
    -----------
    symbols : list
        List of stock symbols 
    start_date : str
        Start date in 'YYYY-MM-DD' format
    end_date : str
        End date in 'YYYY-MM-DD' format
    interval : str, default='1D'
        Time interval for data ('1D' for daily, '1W' for weekly, etc.)
    
    Returns:
    --------
    dict
        Dictionary containing DataFrames for each symbol
    """
    data_dict = {}
    
    for symbol in symbols:
        quote = Quote(symbol=symbol, source="VCI")
        df = quote.history(start=start_date, end=end_date, interval=interval)
        data_dict[symbol] = df
       # Save to CSV with dynamic filename
        filename = f"../data/{symbol}.csv"
        df.to_csv(filename)
        print(f"Saved data for {symbol} to {filename}")
    
    return data_dict

# Example usage:
# symbols = oil_and_gas_symbols
# data = get_stock_data(symbols, "2025-01-01", "2025-05-10")

# Example usage:
# symbols = ["FPT", "VNM", "VCB", "MBB", "VCG", "NTP","HPG", "NTL","TCB", "CSV","NVL", "KDH","PDR", "TCH"]
# data = get_stock_data(symbols, "2023-01-01", "2023-05-10")
def print_symbols_by_industry(df):
    """
    Print stock symbols grouped by their industry classification.
    
    This function takes a DataFrame containing stock information and groups the stock symbols
    by their industry classification. It then prints each industry name followed
    by a comma-separated list of stock symbols belonging to that industry.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing stock information with columns 'industry' and 'stock_symbol'
        
    Returns:
    --------
    None
        Prints the grouped symbols to console
    """
    grouped = df.groupby('industry')['stock_symbol'].apply(list)
    for industry, symbols in grouped.items():
        print(f"{industry}: {', '.join(symbols)}")
        
# print_symbols_by_industry(stock_name)  
def rename_stock_columns(df):
    column_mapping = {
        'symbol': 'stock_symbol',
        'organ_name': 'company_name',
        'icb_name3': 'industry',
        'icb_name2': 'sub_industry_group',
        'icb_name4': 'business_sector',
        'com_type_code': 'company_type',
        'icb_code1': 'industry_code',
        'icb_code2': 'sub_industry_code',
        'icb_code': 'sector_code'
    }
    return df.rename(columns=column_mapping)
# stock_name = rename_stock_columns(stock_name)
