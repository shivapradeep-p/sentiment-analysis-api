import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

# Load datasets
train_df = pd.read_csv('train_data.csv')
test_df = pd.read_csv('test_data.csv')

# Handle missing values (fill NaNs with empty strings)
train_df['text'] = train_df['text'].fillna('')
test_df['text'] = test_df['text'].fillna('')

# Initialize TF-IDF vectorizer
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2))

# Fit on training data and transform
X_train_tfidf = vectorizer.fit_transform(train_df['text'])

# Transform test data
X_test_tfidf = vectorizer.transform(test_df['text'])

# Save vectorizer and transformed data
joblib.dump(vectorizer, 'tfidf_vectorizer.joblib')
joblib.dump(X_train_tfidf, 'X_train_tfidf.joblib')
joblib.dump(X_test_tfidf, 'X_test_tfidf.joblib')

# Confirm shapes
print("Training data TF-IDF shape:", X_train_tfidf.shape)
print("Test data TF-IDF shape:", X_test_tfidf.shape)
