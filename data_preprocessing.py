import pandas as pd
import nltk
import re
import string
from nltk.corpus import stopwords

# Download NLTK resources (only required once)
nltk.download('stopwords')

# Load dataset (ensure the filename matches yours)
df = pd.read_csv('Tweets.csv')

# Keep only relevant columns (sentiment and text)
df = df[['airline_sentiment', 'text']]

# Function to clean text
def clean_text(text):
    text = text.lower()  # Convert text to lowercase
    text = re.sub(r'http\S+|www.\S+', '', text)  # Remove URLs
    text = re.sub(r'@\w+', '', text)  # Remove mentions (@usernames)
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)  # Remove punctuation
    text = re.sub(r'\d+', '', text)  # Remove digits
    text = text.strip()  # Remove extra whitespace
    return text

# Apply cleaning function to text column
df['clean_text'] = df['text'].apply(clean_text)

# Remove stopwords
stop_words = set(stopwords.words('english'))
df['clean_text'] = df['clean_text'].apply(
    lambda x: ' '.join([word for word in x.split() if word not in stop_words])
)

# Preview cleaned data
print("Original Text:\n", df['text'].iloc[0])
print("\nCleaned Text:\n", df['clean_text'].iloc[0])

# Save cleaned dataset for later use
df.to_csv('cleaned_tweets.csv', index=False)
