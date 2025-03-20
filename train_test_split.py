import pandas as pd
from sklearn.model_selection import train_test_split

# Load cleaned dataset
df = pd.read_csv('cleaned_tweets.csv')

# Define features (X) and target labels (y)
X = df['clean_text']
y = df['airline_sentiment']

# Split data into 80% train and 20% test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Confirming the shape of datasets
print("Training set size:", X_train.shape[0])
print("Test set size:", X_test.shape[0])

# Save the split data for future modeling
train_df = pd.DataFrame({'text': X_train, 'sentiment': y_train})
test_df = pd.DataFrame({'text': X_test, 'sentiment': y_test})

train_df.to_csv('train_data.csv', index=False)
test_df.to_csv('test_data.csv', index=False)
