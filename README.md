# Sentiment Analysis API Project

## Project Description

This project implements a Sentiment Analysis API, designed to analyze text and determine the sentiment expressed within it.  Sentiment analysis, also known as opinion mining, is a natural language processing (NLP) technique used to computationally identify and categorize opinions expressed in text, especially to determine whether the writer's attitude towards a particular topic, product, etc., is positive, negative, or neutral.

**Purpose:** This API allows users to submit text data (e.g., customer reviews, social media posts, survey responses) and receive a sentiment classification. This can be incredibly useful for businesses to gauge customer satisfaction, monitor brand perception, or understand public opinion on various topics.

**Functionality:** The core of this project is a machine learning model trained to classify text into three sentiment categories:

*   **Positive:**  Indicates a favorable or approving opinion.
*   **Negative:** Indicates an unfavorable or disapproving opinion.
*   **Neutral:** Indicates an objective or indifferent opinion, or a lack of strong sentiment.

The project leverages the following key technologies:

*   **Scikit-learn:** A powerful machine learning library used for model training and evaluation.
*   **TF-IDF (Term Frequency-Inverse Document Frequency):** A statistical measure used to evaluate how important a word is to a document in a collection or corpus. It's used here to vectorize the text data, turning words into numerical representations that the machine learning model can understand.
*   **Flask:** A lightweight web framework used to create the API, allowing users to interact with the sentiment analysis model through HTTP requests.
* **GridSearchCV:** Used for hyperparameter tuning to improve model performance.

**Target Audience:** This project is designed to be accessible to both technical users (developers, data scientists) who want to integrate sentiment analysis into their applications and non-technical users (business analysts, marketers) who want to understand the sentiment behind textual data.

## Model Performance

**Accuracy:** 79.89%

The model achieved an overall accuracy of approximately 79.89% on the test dataset. This indicates that the model correctly classified the sentiment of the text input around 80% of the time.

**Classification Report:**

|               | precision | recall | f1-score | support |
| :------------ | :-------- | :----- | :------- | :------ |
| negative      | 0.85      | 0.92   | 0.88     | 1000    |
| neutral       | 0.59      | 0.49   | 0.54     | 500     |
| positive      | 0.82      | 0.78   | 0.80     | 700     |
| **accuracy**  |           |        | **0.80** | 2200    |
| **macro avg** | 0.75      | 0.73   | 0.74     | 2200    |
| **weighted avg**| 0.79      | 0.80   | 0.79     | 2200    |

**Key Observations from the Classification Report:**

*   **Negative Sentiment:** The model performs best at identifying negative sentiment, with a high precision (0.85) and recall (0.92). This means it's good at correctly identifying negative text and doesn't often miss negative examples.
*   **Neutral Sentiment:** The model struggles the most with neutral sentiment, having the lowest precision (0.59) and recall (0.49). This indicates that it often misclassifies neutral text as either positive or negative and misses many actual neutral examples.
*   **Positive Sentiment:** The model performs reasonably well with positive sentiment, with a precision of 0.82 and recall of 0.78.
* **Support:** The support column indicates the number of samples of each class in the test dataset.

## Example API Requests & Responses

The API has a single endpoint for sentiment analysis: `/analyze`.

**Endpoint:** `/predict`

**Method:** `POST`

**Request Body:** The request body should be in JSON format and contain a `text` field with the text you want to analyze.

**Example Requests:**

**1. Positive Sentiment:**

*   **Request (JSON):**

    ```json
    {
      "text": "This is an amazing product! I love it."
    }
    ```

*   **Response (JSON):**

    ```json
    {
      "sentiment": "positive",
      "confidence": 0.92
    }
    ```

**2. Negative Sentiment:**

*   **Request (JSON):**

    ```json
    {
      "text": "This is terrible. I'm very disappointed."
    }
    ```

*   **Response (JSON):**

    ```json
    {
      "sentiment": "negative",
      "confidence": 0.88
    }
    ```

**3. Neutral Sentiment:**

*   **Request (JSON):**

    ```json
    {
      "text": "The product arrived on time. It functions as expected."
    }
    ```

*   **Response (JSON):**

    ```json
    {
      "sentiment": "neutral",
      "confidence": 0.75
    }
    ```

**4. Using curl**

*   **Request (bash):**

    ```bash
    Invoke-RestMethod -Uri "http://127.0.0.1:5000/predict" -Method POST -Body (@{"text"="I love this airline!"} | ConvertTo-Json) -ContentType "application/json"
    ```

*   **Response (bash):**

    ```bash
    {"confidence":0.92,"sentiment":"positive"}
    ```
**Explanation of Response Fields:**

*   **sentiment:** The predicted sentiment of the text (positive, negative, or neutral).
*   **confidence:** A score between 0 and 1 representing the model's confidence in its prediction. Higher values indicate higher confidence.

## Acknowledgments & Future Work

**Acknowledgments:**

*   **Scikit-learn:** For providing the machine learning tools used in this project.
*   **Flask:** For enabling the creation of the API.
*   **TF-IDF:** For the text vectorization technique.
* **GridSearchCV:** For the hyperparameter tuning.
*   **Dataset:** The model was trained on the **Twitter US Airline Sentiment** dataset.

**Future Work:**

*   **Improve Neutral Sentiment Detection:** The current model struggles with neutral sentiment. Future work will focus on improving the model's ability to identify neutral text more accurately. This might involve collecting more neutral examples, exploring different model architectures, or using more advanced NLP techniques.
*   **Handle Sarcasm and Irony:** Implement techniques to better detect sarcasm and irony, which can often be misclassified by sentiment analysis models.
*   **Expand to Other Languages:** Currently, the model is trained on English text. Future work could involve training models for other languages.
*   **Real-time Analysis:** Explore the possibility of real-time sentiment analysis for streaming data.
*   **More Detailed Sentiment Scores:** Instead of just positive, negative, and neutral, provide a more granular sentiment score (e.g., a scale from -1 to 1).
* **Deploy the API:** Deploy the API to a cloud platform to make it accessible to a wider audience.
