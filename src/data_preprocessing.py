import os
import pandas as pd

def load_and_preprocess(news_file, stock_folder):
    """
    Load and preprocess news and stock data.
    
    Parameters:
        news_file (str): Path to the news data CSV file.
        stock_folder (str): Path to the folder containing stock data CSV files.
        
    Returns:
        news_data (DataFrame): Preprocessed news data.
        stock_data (dict): A dictionary of stock data DataFrames indexed by stock symbols.
    """
    # Load news data
    print("Loading news data...")
    news_data = pd.read_csv(news_file)
    
    # Normalize the Date column by stripping timezone info and retaining the date part
    news_data['Date'] = pd.to_datetime(news_data['Date'].str.split(' ').str[0], errors='coerce', utc=True)

    # Drop rows with missing dates or headlines
    invalid_rows = news_data['Date'].isna().sum()
    print(f"Dropping {invalid_rows} rows with invalid dates.")
    news_data = news_data.dropna(subset=['Date', 'headline'])
    
    print("News data loaded and preprocessed successfully!")
    
    # Load stock data from multiple CSV files
    stock_data = {}
    print("Loading stock data...")
    for file in os.listdir(stock_folder):
        if file.endswith('.csv'):
            stock_symbol = file.split('.')[0]  # Extract stock symbol from filename
            file_path = os.path.join(stock_folder, file)
            print(f"Loading stock data for {stock_symbol}...")
            stock_df = pd.read_csv(file_path)
            
            # Normalize the stock Date column by stripping timezone info and retaining the date part
            stock_df['Date'] = pd.to_datetime(stock_df['Date'].str.split(' ').str[0], errors='coerce', utc=True)
            stock_df = stock_df.dropna(subset=['Date'])  # Drop rows with invalid dates
            
            stock_data[stock_symbol] = stock_df
            print(f"Loaded stock data for {stock_symbol}")
    
    print("Stock data loaded and preprocessed successfully!")
    return news_data, stock_data
