import pandas as pd
import matplotlib.pyplot as plt
from textblob import TextBlob
from sklearn.feature_extraction.text import CountVectorizer

# Load the data
try:
    news_data = pd.read_csv('../raw_data/raw_analyst_ratings.csv')  # Adjust path if necessary
    print("Dataset successfully loaded!")
except FileNotFoundError:
    print("Error: The file 'raw_analyst_ratings.csv' was not found. Check the file path.")
    exit()

# Check for missing values in the 'publisher' column
if 'publisher' not in news_data.columns:
    print("Error: The dataset does not contain a 'publisher' column.")
    exit()

if news_data['publisher'].isnull().any():
    print("Warning: Some articles are missing publisher information.")
    # Fill missing publishers with 'Unknown' (optional)
    news_data['publisher'].fillna('Unknown', inplace=True)

# Normalize publisher names for consistency
news_data['publisher'] = news_data['publisher'].str.strip().str.lower()

# 1. Descriptive Statistics
# Headline Length Analysis
if 'headline' in news_data.columns:
    news_data['headline_length'] = news_data['headline'].str.len()
    print("\nDescriptive Statistics for Headline Lengths:")
    print(news_data['headline_length'].describe())
else:
    print("Error: The dataset does not contain a 'headline' column.")

# Publisher Activity (Count Articles Per Publisher)
article_counts = news_data['publisher'].value_counts()
print("\nArticles per Publisher:")
print(article_counts)

# Visualize Publisher Activity
plt.figure(figsize=(10, 6))  # Set figure size for better readability
top_publishers = article_counts.head(10)  # Display top 10 publishers
top_publishers.plot(kind='bar', title='Top 10 Publishers by Article Count')
plt.xlabel('Publisher')
plt.ylabel('Number of Articles')
plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better visibility
plt.tight_layout()  # Adjust layout to prevent clipping
plt.show()

# 2. Publication Trends Over Time
if 'date' in news_data.columns:
    news_data['date'] = pd.to_datetime(news_data['date'], errors='coerce')
    publication_trends = news_data['date'].dt.date.value_counts().sort_index()

    print("\nPublication Trends Over Time:")
    print(publication_trends)

    # Plot publication frequency over time
    publication_trends.plot(kind='line', title='Publication Trends Over Time', figsize=(10, 6))
    plt.xlabel('Date')
    plt.ylabel('Number of Articles')
    plt.show()
else:
    print("Error: The dataset does not contain a 'date' column.")

# 3. Text Analysis
# Sentiment Analysis
if 'headline' in news_data.columns:
    news_data['sentiment'] = news_data['headline'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
    news_data['sentiment_label'] = news_data['sentiment'].apply(
        lambda x: 'positive' if x > 0 else 'negative' if x < 0 else 'neutral'
    )

    print("\nSentiment Analysis Results:")
    print(news_data['sentiment_label'].value_counts())

    # Visualize sentiment distribution
    news_data['sentiment_label'].value_counts().plot(kind='bar', title='Sentiment Distribution')
    plt.show()
else:
    print("Error: The dataset does not contain a 'headline' column.")

# Topic Modeling (Common Keywords)
if 'headline' in news_data.columns:
    vectorizer = CountVectorizer(max_features=10, stop_words='english')
    X = vectorizer.fit_transform(news_data['headline'].dropna())
    print("\nTop Keywords:")
    print(vectorizer.get_feature_names_out())
else:
    print("Error: The dataset does not contain a 'headline' column.")

# 4. Publisher Analysis
# Extracting domains from email-based publisher names
if 'publisher' in news_data.columns:
    news_data['publisher_domain'] = news_data['publisher'].str.extract(r'@([a-zA-Z0-9.-]+)')[0]
    print("\nPublisher Domains:")
    print(news_data['publisher_domain'].value_counts())
else:
    print("Error: The dataset does not contain a 'publisher' column.")

# Save publisher counts to CSV (optional)
try:
    article_counts.to_csv('articles_per_publisher.csv', header=True)
    print("Article counts saved to 'articles_per_publisher.csv'.")
except Exception as e:
    print(f"Error saving to CSV: {e}")
