# Online News Popularity Prediction

## Project Overview
This project explores and predicts the popularity of online news articles using machine learning techniques. By analyzing a dataset of articles from Mashable.com, we aim to identify key factors that influence article engagement and develop models to predict the number of shares an article will receive.

## Dataset
The dataset contains metadata from 39,797 articles published on Mashable.com, with 61 attributes including:
- Content features (word count, images, links)
- Keyword metrics
- Natural language processing metrics (sentiment, subjectivity)
- Temporal features (day of week, weekend)
- Social metrics (shares, author followers)

## Methodology

### Exploratory Data Analysis
- Analyzed class distribution of article shares
- Examined content, channel, and author-related features
- Identified temporal patterns in publishing and engagement
- Performed correlation analysis to find predictive features

### Feature Engineering
- Transformed skewed features to improve distribution
- Applied standardization to numeric features
- Created interaction features and polynomial features
- Reduced dimensionality by removing highly correlated features

### Modeling
- Applied SMOTE to handle class imbalance
- Developed and compared multiple classification models:
  - Logistic Regression
  - Random Forest
  - Gradient Boosting
  - XGBoost
  - LightGBM
  - Neural Networks
- Created ensemble models through stacking
- Performed hyperparameter tuning for optimal performance
- Evaluated with accuracy, F1-score and confusion matrices

## Key Findings
- Weekends show higher sharing rates despite fewer articles being published
- Certain content channels (Tech, Social Media) correlate with higher shares
- Keyword metrics strongly influence article shareability
- Best model achieved ~68% accuracy in predicting share categories
- Important features include: keyword metrics, whether published on weekend, channel type, and sentiment measures

## Models Evaluated
| Model | Test Accuracy | F1 Score |
|-------|--------------|----------|
| Optimized XGBoost | 67.84% | 67.85% |
| Top 3 Stacking | 67.51% | 67.55% |
| LightGBM | 67.39% | 67.35% |
| Gradient Boosting | 67.00% | 67.01% |
| Random Forest | 66.97% | 66.99% |
| Logistic Regression | 66.41% | 66.34% |
| XGBoost | 66.12% | 66.11% |
| Neural Network | 61.45% | 61.46% |

## Challenges and Future Work
- Limited predictive accuracy suggests external factors influence sharing behavior
- Feature engineering and hyperparameter optimization provided modest gains
- Future approaches could incorporate real-time social media data
- Integrating audience demographics and browsing behavior could improve prediction accuracy

## Team Members
Anqi Gu, Evelyn Zhou, Han Zhang, Hongyu Liao, Yiwei Li

## Course
MSDS 422 Practical Machine Learning (March 2025)
