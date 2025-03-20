import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

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

# Define hyperparameters for tuning
param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],         # Regularization strength
    'solver': ['liblinear', 'lbfgs'],     # Optimization algorithm
    'penalty': ['l1', 'l2']               # Regularization type
}

# GridSearchCV setup
grid_search = GridSearchCV(classifier, param_grid, cv=5, scoring='accuracy', n_jobs=-1)

# Perform Grid Search on training data
print("Performing Grid Search...")
grid_search.fit(X_train, y_train)

# Best hyperparameters
print("\nBest Hyperparameters:", grid_search.best_params_)

# Evaluate the tuned model on test data
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

# Classification report after tuning
print("\nClassification Report (after tuning):\n", classification_report(y_test, y_pred))

# Save the tuned model
joblib.dump(best_model, 'logistic_regression_tuned.joblib')
