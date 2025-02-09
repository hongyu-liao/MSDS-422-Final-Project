# Online News Popularity Data Analysis and Feature Engineering

This project uses the [Online News Popularity](https://archive.ics.uci.edu/ml/datasets/online+news+popularity) dataset from the UCI Machine Learning Repository to analyze the factors that drive the popularity (shares) of news articles.

## 1. Dataset Introduction

The Online News Popularity dataset contains metadata and social feedback (shares) for 39,797 news articles published on Mashable. Each sample has 61 attributes, including:

* **Non-predictive attributes (2):** `url` (article link) and `timedelta` (days between article publication and dataset collection).
* **Predictive attributes (59):** Various features such as number of words, links, keywords, subject category, sentiment polarity, day of the week, and results from LDA topic modeling.

The target variable is `shares`, representing the number of times the article was shared on social media.

## 2. Exploratory Data Analysis (EDA)

This section details the steps taken to understand the dataset's underlying patterns, distributions, and potential anomalies.

### 2.1. Data Loading and Preliminary Analysis
- Data is loaded using Pandas, and initial dataset information, head, descriptive statistics, and missing values are evaluated.

### 2.2. Visualization of Distributions
- The distribution of the target variable `shares` is visualized.
- A log-transformation (`log_shares`) is applied to mitigate skewness.
- A Box-Cox transformation further normalizes the `shares` distribution.
- Histograms and boxplots are used to assess data spread and detect outliers.

### 2.3. Correlation Analysis
- A correlation matrix of numerical features is computed and visualized using a heatmap.
- The most correlated feature pairs are identified for further analysis.

### 2.4. Outlier Detection and Analysis
- Outliers are detected using both the IQR method and Z-score analysis.
- Visualizations demonstrate the effect of weekdays and data channels on article shares.

## 3. Feature Engineering

Feature engineering steps were applied to enrich the dataset and improve model performance.

### 3.1. Data Cleaning and Boolean Conversion
- Irrelevant attributes such as `url` and `timedelta` are removed.
- Binary indicators (e.g., weekdays and data channels) are converted to boolean types.

### 3.2. New Feature Creation
- New features such as `log_shares`, `boxcox_shares`, and `shares_category` are engineered.
- `shares_category` segments articles into 'Low', 'Medium-Low', 'Medium-High', and 'High' popularity groups.
- Feature skewness is addressed using transformations (Yeo-Johnson, square root, inverse) based on statistical distribution.

### 3.3. Visualization and Distribution Analysis
- Feature distributions are analyzed through histograms, count plots, and bar charts.
- Group-wise comparisons by `shares_category` reveal key differences in feature behaviors.

## 4. Summary

The EDA and feature engineering steps provide valuable insights into the factors influencing the popularity of news articles. These analyses lay the foundation for building predictive models.

## 5. File Structure and Execution

- **Data Files:** `OnlineNewsPopularity.csv`, `OnlineNewsPopularity.names`
- **Jupyter Notebook:** `EDA and Feature Engineering.ipynb`
- **Documentation:** This README contains an overview of the analysis and feature engineering processes.

*Note: For a complete and reproducible analysis, please refer to the Jupyter Notebook.*
