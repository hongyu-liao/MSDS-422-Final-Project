# Online News Popularity Data Analysis and Feature Engineering

This project uses the [Online News Popularity](https://archive.ics.uci.edu/ml/datasets/online+news+popularity) dataset from the UCI Machine Learning Repository to perform Exploratory Data Analysis (EDA) and feature engineering. The goal is to predict the popularity (number of shares) of news articles.

## 1. Dataset Introduction

The Online News Popularity dataset contains metadata and social feedback (shares) for 39,797 news articles published on Mashable. Each sample has 61 attributes, including:

*   **Non-predictive attributes (2):** `url` (article link) and `timedelta` (days between article publication and dataset collection).
*   **Predictive attributes (59):**  These cover various features of the articles, such as:
    *   Number of words
    *   Number of links
    *   Keywords
    *   Subject category
    *   Sentiment polarity
    *   Day of the week
    *   Results from LDA (Latent Dirichlet Allocation) topic modeling
    *   ...

The target variable is `shares`, representing the number of times the article was shared on social media.

## 2. Exploratory Data Analysis (EDA)

This section explores the data to discover patterns, relationships, and potential issues.

### 2.1. Data Loading and Preview

```python
# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
data = pd.read_csv('OnlineNewsPopularity.csv')

# Display the first few rows
print(data.head())

# View data information
print(data.info())

# View descriptive statistics
print(data.describe())
