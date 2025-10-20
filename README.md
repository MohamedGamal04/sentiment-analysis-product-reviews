# Sentiment Analysis on Product Reviews

This project performs sentiment analysis on a dataset of 50,000 movie reviews from IMDB. The goal is to classify the reviews as positive or negative using various machine learning techniques.

## Project Structure

```
sentiment-analysis-product-reviews
├── data
│   ├── raw
│   │   └── IMDB Dataset.csv         # Raw dataset of movie reviews
│   └── processed                     # Directory for processed data
├── notebooks
│   └── Sentiment Analysis on Product Reviews.ipynb  # Jupyter notebook for analysis
├── src
│   ├── __init__.py                  # Marks the src directory as a package
│   ├── data_processing.py           # Functions for loading and processing data
│   ├── preprocessing.py              # Text preprocessing functions
│   ├── features.py                   # Feature extraction methods
│   ├── models.py                     # Machine learning models for classification
│   └── visualization.py              # Visualization functions for results
├── requirements.txt                  # Required Python packages
├── .gitignore                        # Files and directories to ignore in Git
├── README.md                         # Project documentation
└── LICENSE                           # Licensing information
```

## Installation

To set up the project, clone the repository and install the required packages:

```bash
git clone <repository-url>
cd sentiment-analysis-product-reviews
pip install -r requirements.txt
```

## Usage

1. Load the dataset and perform initial analysis using the Jupyter notebook located in the `notebooks` directory.
2. Use the functions in the `src` directory to preprocess the data, extract features, and train machine learning models.
3. Visualize the results using the provided visualization functions.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.

## Acknowledgments

- The dataset used in this project is sourced from [IMDB](https://www.imdb.com/).
- Special thanks to the contributors and libraries that made this project possible.