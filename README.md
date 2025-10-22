# Sentiment Analysis on Product Reviews
This project performs sentiment analysis on a dataset of 50,000 movie reviews from IMDB. The goal is to classify the reviews as positive or negative using various machine learning techniques.

## Requirements (recommended)
- Python 3.8+
- numpy, pandas, scikit-learn
- nltk, num2words, contractions
- plotly

Installation (Windows PowerShell):
1. Create & activate venv
   python -m venv .venv
   .venv\Scripts\Activate.ps1

2. Install packages
   pip install -r requirements.txt

If you don't have a requirements.txt, install core packages:
   pip install numpy pandas scikit-learn nltk num2words contractions plotly

Run required NLTK downloads in the notebook (cells include downloads for punkt, stopwords, wordnet, averaged_perceptron_tagger).

## Acknowledgments
- The dataset used in this project is sourced from [Kaggle](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews).
