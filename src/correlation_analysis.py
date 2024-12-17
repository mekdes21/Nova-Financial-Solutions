import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def compute_daily_returns(stock_data):
    """
    Calculate daily returns for each stock in the stock_data dictionary.
    """
    for stock_ticker, stock_df in stock_data.items():
        stock_df['daily_return'] = stock_df['Close'].pct_change()
        stock_data[stock_ticker] = stock_df
    return stock_data

def calculate_correlation(news_data, stock_data):
    """
    Calculate the correlation between sentiment scores in news_data 
    and daily stock returns in stock_data.
    """
    correlations = {}

    # Ensure column consistency
    news_data.columns = news_data.columns.str.strip()

    for stock_ticker, stock_df in stock_data.items():
        stock_df.columns = stock_df.columns.str.strip()

        # Check for the 'Date' column
        if 'Date' not in stock_df.columns or 'Date' not in news_data.columns:
            raise KeyError(f"Missing 'Date' column in news_data or stock_data for {stock_ticker}.")

        # Ensure the 'Date' columns are in datetime format
        news_data['Date'] = pd.to_datetime(news_data['Date'], errors='coerce')
        stock_df['Date'] = pd.to_datetime(stock_df['Date'], errors='coerce')

        # Drop rows where 'Date' is invalid
        news_data = news_data.dropna(subset=['Date'])
        stock_df = stock_df.dropna(subset=['Date'])

        # Merge news_data and stock_df on 'Date'
        merged_data = pd.merge(news_data, stock_df[['Date', 'daily_return']], on='Date', how='inner')

        # Ensure 'sentiment_score' exists
        if 'sentiment_score' in merged_data.columns:
            correlation = merged_data['sentiment_score'].corr(merged_data['daily_return'])
            correlations[stock_ticker] = correlation
        else:
            print(f"Missing 'sentiment_score' in merged data for {stock_ticker}. Skipping.")

    return correlations


def plot_correlations(correlations):
    """
    Plot correlations between sentiment and stock returns.
    """
    # Convert the correlations dictionary to a DataFrame for plotting
    correlation_df = pd.DataFrame(list(correlations.items()), columns=['Stock', 'Correlation'])

    # Plot the correlations using seaborn
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Stock', y='Correlation', data=correlation_df, palette='viridis')
    plt.title("Correlation Between Sentiment and Stock Returns")
    plt.xlabel('Stock Ticker')
    plt.ylabel('Correlation')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Example usage
    from data_preprocessing import load_and_preprocess
    from sentiment_analysis import perform_sentiment_analysis

    # Load the data
    news_data, stock_data = load_and_preprocess(
        'C:/Users/Hp/Nova-Financial-Solutions/raw_data/raw_analyst_ratings.csv',
        'C:/Users/Hp/Nova-Financial-Solutions/raw_data/yfinance_data/'
    )

    # Perform sentiment analysis
    news_data = perform_sentiment_analysis(news_data)

    # Compute daily stock returns
    stock_data = compute_daily_returns(stock_data)

    # Calculate correlations
    correlations = calculate_correlation(news_data, stock_data)

    # Display results
    for stock_ticker, correlation in correlations.items():
        print(f"Correlation between sentiment and stock returns for {stock_ticker}: {correlation}")

    # Plot correlations
    plot_correlations(correlations)
