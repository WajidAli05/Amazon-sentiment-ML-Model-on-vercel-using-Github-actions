import pandas as pd
import numpy as np
import re
import nltk
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import pickle
import warnings
warnings.filterwarnings("ignore")

# Load the dataset from the same directory
df = pd.read_csv('amazon.csv')  # Ensure the amazon.csv file is in the same folder as this script

# Inspect the first few rows (optional)
print(df.head())

# Preprocess the data
# Fill missing values in 'reviewText' with empty strings
df['reviewText'] = df['reviewText'].fillna('')

# Clean the text (remove numbers and special characters)
df['cleaned_review'] = df['reviewText'].apply(lambda review: re.sub("[^a-zA-Z]", ' ', review).lower())

# Initialize SentimentIntensityAnalyzer
sid = SentimentIntensityAnalyzer()

# Apply sentiment analysis using VADER to label positive/negative sentiments
df['compound'] = df['cleaned_review'].apply(lambda x: sid.polarity_scores(x)['compound'])
df['sentiment'] = df['compound'].apply(lambda x: 'positive' if x >= 0 else 'negative')

# Drop rows without review text (if any)
df = df[df['cleaned_review'].str.strip() != '']

# Define features and target
X = df['cleaned_review']  # Reviews
y = df['sentiment']       # Sentiment labels (positive/negative)

# Split the dataset into training and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Use TF-IDF vectorization to convert text into numerical form
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Initialize Logistic Regression model
model = LogisticRegression()

# Train the model on the training data
model.fit(X_train_tfidf, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test_tfidf)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")
print(classification_report(y_test, y_pred))

# Save the trained model and vectorizer as files
model_filename = 'sentiment_model.pkl'
vectorizer_filename = 'tfidf_vectorizer.pkl'

# Save the model to a file
with open(model_filename, 'wb') as model_file:
    pickle.dump(model, model_file)

# Save the vectorizer to a file
with open(vectorizer_filename, 'wb') as vectorizer_file:
    pickle.dump(vectorizer, vectorizer_file)

print(f"Model saved as: {model_filename}")
print(f"Vectorizer saved as: {vectorizer_filename}")
#ePoFS6T7WpMVLMBYAZUE5SXi