import pandas as pd
import talib
import matplotlib.pyplot as plt
import os

# Define the function to calculate technical indicators (e.g., Simple Moving Average - SMA)
def calculate_technical_indicators(stock_data):
    # Calculate the 50-period Simple Moving Average (SMA)
    stock_data['SMA_50'] = talib.SMA(stock_data['Close'], timeperiod=50)
    return stock_data

# Define the function to plot the technical indicators (Close price and SMA)
def plot_technical_indicators(stock_data, stock_symbol):
    plt.figure(figsize=(10, 6))
    plt.plot(stock_data['Date'], stock_data['Close'], label=f'{stock_symbol} Close Price', color='blue')
    plt.plot(stock_data['Date'], stock_data['SMA_50'], label=f'{stock_symbol} 50-period SMA', color='red')
    plt.title(f'{stock_symbol} Stock Price and 50-period Simple Moving Average')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Folder where stock data CSV files are stored
data_folder = '../raw_data/'  # Adjust path if necessary

# List of stock symbols and corresponding file paths
stocks = [
    ('TSLA', 'TSLA_historical_data.csv'),
    ('NVDA', 'NVDA_historical_data.csv'),
    ('MSFT', 'MSFT_historical_data.csv'),
    ('META', 'META_historical_data.csv'),
    ('GOOG', 'GOOG_historical_data.csv'),
    ('AMZN', 'AMZN_historical_data.csv'),
    ('AAPL', 'AAPL_historical_data.csv')
]

# Loop through each stock file, apply technical indicator analysis, and plot results
for stock_symbol, file_name in stocks:
    # Construct the full file path
    file_path = os.path.join(data_folder, file_name)
    
    # Read the stock data from CSV file
    stock_data = pd.read_csv(file_path)
    
    # Convert the 'Date' column to datetime format
    stock_data['Date'] = pd.to_datetime(stock_data['Date'])
    
    # Calculate the technical indicators (SMA in this case)
    stock_data = calculate_technical_indicators(stock_data)
    
    # Plot the technical indicators (Stock price and SMA)
    plot_technical_indicators(stock_data, stock_symbol)
