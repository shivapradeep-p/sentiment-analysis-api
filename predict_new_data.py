import joblib

# Load the trained model and vectorizer
model = joblib.load('logistic_regression_tuned.joblib')
vectorizer = joblib.load('tfidf_vectorizer.joblib')

# Define new customer feedback examples
new_reviews = [
    "The flight was amazing! The crew was so kind and helpful.",
    "Worst experience ever. Flight delayed for 5 hours and no compensation.",
    "The service was okay, nothing special.",
    "I love the new in-flight entertainment system. Best experience!",
    "Horrible food and rude staff. Never flying with them again."
]

# Convert new text data to TF-IDF features
new_reviews_tfidf = vectorizer.transform(new_reviews)

# Predict sentiment
predictions = model.predict(new_reviews_tfidf)

# Display results
for review, sentiment in zip(new_reviews, predictions):
    print(f"Review: {review}\nPredicted Sentiment: {sentiment}\n")
