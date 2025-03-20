import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load TF-IDF data
X_train = joblib.load('X_train_tfidf.joblib')
X_test = joblib.load('X_test_tfidf.joblib')

# Load labels
train_df = pd.read_csv('train_data.csv')
test_df = pd.read_csv('test_data.csv')

y_train = train_df['sentiment']
y_test = test_df['sentiment']

# Initialize Logistic Regression
classifier = LogisticRegression(max_iter=1000, random_state=42)

# Train the model
classifier.fit(X_train, y_train)

# Predict on test data
y_pred = classifier.predict(X_test)

# Evaluate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy on test data:", accuracy)

# Detailed classification report
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Save the trained model for later use
joblib.dump(classifier, 'logistic_regression_model.joblib')
