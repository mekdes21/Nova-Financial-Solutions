import pandas as pd
import matplotlib.pyplot as plt

# Load the data
news_data = pd.read_csv('../raw_data/raw_analyst_ratings.csv')  # Adjust path if necessary

# Count articles per publisher
article_counts = news_data['publisher'].value_counts()

# Print the result
print("Articles per Publisher:")
print(article_counts)

# Optional: Visualize the result
article_counts.plot(kind='bar', title='Articles per Publisher')
plt.xlabel('Publisher')
plt.ylabel('Number of Articles')
plt.show()

# Save to CSV (optional)
article_counts.to_csv('articles_per_publisher.csv', header=True)
