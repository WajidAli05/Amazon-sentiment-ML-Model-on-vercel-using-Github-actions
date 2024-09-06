from fastapi import FastAPI
import pickle
from pydantic import BaseModel
from typing import List

# Define the FastAPI app
app = FastAPI()

# Load the trained model and vectorizer
with open("sentiment_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("tfidf_vectorizer.pkl", "rb") as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# Define the input structure
class ReviewInput(BaseModel):
    reviews: List[str]

# API root endpoint
@app.get("/")
def read_root():
    return {"message": "Sentiment analysis model is live"}

# API endpoint for sentiment prediction
@app.post("/predict")
def predict_sentiment(data: ReviewInput):
    reviews = data.reviews
    transformed_reviews = vectorizer.transform(reviews)
    predictions = model.predict(transformed_reviews)
    return {"predictions": list(predictions)}
