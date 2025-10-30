# Sentiment Analysis on Product Reviews

This project implements a sentiment analysis system for product reviews using machine learning techniques. It focuses on classifying reviews as positive or negative based on their textual content with a Logistic Regression model trained on a large dataset of movie reviews.

## Features

- Text preprocessing including contraction expansion, stopword removal, lemmatization, and number conversion
- TF-IDF feature extraction from reviews
- Logistic Regression classification for sentiment prediction
- Exploratory data analysis and visualization
- Batch prediction support
- Streamlit UI demo for interactive sentiment analysis

## Project Structure

- `model.pkl` `vectorizer.pkl` - model and vectorizer files
- `Sentiment Analysis on Product Reviews.ipynb` - Jupyter Notebook containing preprocessing, training, and evaluation pipeline
- `app.py` - Streamlit application script for live sentiment analysis (create separately)
- `requirements.txt` - Required Python packages
- `README.md` - Project overview and usage instructions

## Installation

1. Clone the repository:
``` powershell
git clone https://github.com/MohamedGamal04/sentiment-analysis-product-reviews.git
cd sentiment-analysis-product-reviews
```
2. Create a Python environment and activate it (recommended):
``` powershell
conda create -n sentiment-env python=3.8
conda activate sentiment-env
```
3. Install dependencies:
``` powershell
pip install -r requirements.txt
```
4. Download necessary NLTK data resources:
``` python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
```
## Usage

- Train the model and evaluate using the Jupyter Notebook.
- To use the Streamlit app (once created and set up), run:
``` terminal
streamlit run app.py
```

- The Streamlit UI allows you to enter reviews manually or upload CSV files for batch prediction.

## Dataset

This project uses the [IMDB movie reviews dataset](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews) containing 50,000 labeled reviews for training and testing the sentiment classifier .


