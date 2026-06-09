# BBC News Prediction - Text Classification

A machine learning project that classifies BBC news articles into different categories using Support Vector Machine (SVM) with TF-IDF vectorization.

## Project Overview

This project demonstrates text classification using natural language processing techniques. It trains an SVM model on BBC news articles to predict which category (Politics, Business, Sport, etc.) a given news headline or article belongs to.

## Dataset

- **File**: `bbc_text_cls new.csv`
- **Features**: Text content of news articles
- **Target**: News categories/labels
- **Format**: CSV with columns for text and labels

## Technologies & Libraries

- **Python 3.10+**
- **scikit-learn**: Machine learning library for SVM and text vectorization
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **Jupyter Notebook**: Interactive development environment

## Features

- **TF-IDF Vectorization**: Converts text into numerical features
  - 50,000 maximum features
  - Unigrams and bigrams (1-2 word combinations)
  - English stop words removal
  - Sublinear TF scaling

- **Support Vector Machine (SVM)**
  - RBF kernel
  - C parameter: 10
  - Gamma: scale (automatically determined)
  - Probability estimates enabled

## Model Performance

The model is evaluated using:
- **Accuracy**: Overall correct predictions
- **F1 Score**: Weighted harmonic mean of precision and recall
- **Classification Report**: Per-class metrics including precision, recall, and F1-score

## Usage

### 1. Prerequisites

Install required packages:
```bash
pip install pandas scikit-learn numpy jupyter
```

### 2. Running the Project

Open the Jupyter notebook:
```bash
jupyter notebook BBC_news_prediction.ipynb
```

### 3. Training the Model

The notebook includes:
- Data loading from CSV
- Text vectorization with TF-IDF
- Train-test split (80-20)
- SVM model training
- Performance evaluation

### 4. Making Predictions

Example of predicting a new headline:
```python
new_headline = "he is elected as the new mayor"
headline_vector = vectorizer.transform([new_headline])
prediction = model.predict(headline_vector)
print(f"Prediction: {prediction}")
```

## File Structure

```
BBC_news_prediction/
├── README.md                      # Project documentation
├── BBC_news_prediction.ipynb      # Main notebook with model implementation
├── bbc_text_cls new.csv           # Dataset with news articles and labels
└── .gitignore                     # Git ignore file
```

## Model Training Pipeline

1. **Data Loading**: Load CSV file into pandas DataFrame
2. **Vectorization**: Convert text to TF-IDF feature vectors
3. **Data Splitting**: Split into 80% training and 20% testing
4. **Model Training**: Train SVM with RBF kernel
5. **Prediction**: Generate predictions on test set
6. **Evaluation**: Calculate accuracy, F1-score, and classification metrics

## Results

The model outputs:
- Accuracy score (0-1 scale)
- Weighted F1 score
- Detailed classification report with per-class metrics
- Sample predictions with actual vs predicted values

## Example Output

```
Accuracy: [model accuracy percentage]
F1 score: [weighted F1 score]

Classification Report:
              precision    recall  f1-score   support
      [class]       ...       ...       ...       ...
```

## Key Hyperparameters

| Parameter | Value | Purpose |
|-----------|-------|---------|
| test_size | 0.2 | 20% data for testing |
| random_state | 42 | Reproducibility |
| max_features | 50000 | Max TF-IDF features |
| ngram_range | (1, 2) | Unigrams and bigrams |
| SVM C | 10 | Regularization strength |
| SVM kernel | rbf | Non-linear classification |

## Future Enhancements

- Experiment with different kernels (linear, poly)
- Hyperparameter tuning with GridSearchCV
- Use deep learning models (LSTM, BERT)
- Add preprocessing (lemmatization, stemming)
- Implement cross-validation for better evaluation
- Add model persistence (pickle/joblib)

## Author

Created for machine learning education and practice

## License

This project is open source and available under the MIT License.

## References

- [scikit-learn Documentation](https://scikit-learn.org/)
- [TF-IDF Vectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)
- [SVM Classifier](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)
