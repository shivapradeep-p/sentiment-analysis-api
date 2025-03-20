from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Load model and vectorizer
model = joblib.load('logistic_regression_tuned.joblib')
vectorizer = joblib.load('tfidf_vectorizer.joblib')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json  # Expecting a JSON request with "text"
    
    if 'text' not in data:
        return jsonify({'error': 'No text provided'}), 400
    
    text = [data['text']]  # Wrap input as a list
    text_tfidf = vectorizer.transform(text)
    prediction = model.predict(text_tfidf)[0]  # Get prediction

    return jsonify({'text': data['text'], 'sentiment': prediction})

if __name__ == '__main__':
    app.run(debug=True)
