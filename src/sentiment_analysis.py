import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

def perform_sentiment_analysis(news_data):
    # Ensure the 'headline' column exists
    if 'headline' not in news_data.columns:
        raise ValueError("The 'headline' column is missing from the news data.")
    
    analyzer = SentimentIntensityAnalyzer()

    # Add a column for sentiment scores using VADER
    news_data['sentiment_score'] = news_data['headline'].apply(lambda x: analyzer.polarity_scores(str(x))['compound'])

    # Aggregate sentiment by date (average sentiment per day)
    daily_sentiment = news_data.groupby('Date')['sentiment_score'].mean().reset_index()

    return daily_sentiment

# Example usage
#news_data = pd.read_csv('C:/Users/Hp/Nova-Financial-Solutions/raw_data/raw_analyst_ratings.csv')  
#sentiment_data = perform_sentiment_analysis(news_data)
#print(sentiment_data.head())
