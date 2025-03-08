# Online News Popularity Data Analysis

This project uses the [Online News Popularity](https://archive.ics.uci.edu/ml/datasets/online+news+popularity) dataset from the UCI Machine Learning Repository to analyze the factors that drive the popularity (shares) of news articles.

# Executive Summary

## Background:

The exponential growth in online news consumption has created a complex landscape for content creators and publishers. Through the analysis of the Online News Popularity dataset, this research aims to uncover the underlying factors that influence article sharing behavior and develop a robust predictive model using ML/DL techniques. The findings will provide data-driven insights to optimize content strategy and help content creators to predict the popularity of their articles.

## Objectives:

The primary objectives of this project are to:

1. Identify the key features in the Online News Popularity dataset that are significantly correlated with the number of article shares.
2. Analyze the impact of various features (like sentiment polarity, keywords, publication timing) on article sharing.
3. Build multiple classification models that can accurately forecast the group of shares an article will receive based on its features.
4. Evaluate different models and compare their performance.


## Methods:

We utilized the Online News Popularity dataset from the UCI Machine Learning Repository, which contains data on 39,797 articles published on Mashable. Our analysis involved Exploratory Data Analysis (EDA) to understand data patterns and relationships, followed by feature engineering to handle feature that can help improve the performance of the model. Then, we built multiple models including XGBoost, Random Forest, Neural Network (not yet decided). We evaluated the performance of the models using R-squared, Mean Absolute Error (MAE), Mean Squared Error (MSE), and Root Mean Squared Error (RMSE).


# Problem Statement

The explosion of online news sources and the growing dependence on social media for news consumption have created a highly competitive environment for content creators and publishers. Predicting the popularity of online news articles, especially the number of times an article is shared on social media, is a big challenge. The complex interaction of various factors influencing an article's virality, making the prediction difficult. This project aims to address this challenge by analyzing the Online News Popularity dataset to identify key predictors of article shares and develop a useful classification model. 

# Research Objectives

To address the problem outlined above, this project aims to achieve the following specific research objectives:

1. **Identify Key Features:** Determine which features from the Online News Popularity dataset show statistically significant relationship with target variable `shares`.
2. **Quantify Feature Influence:** Analyze the influence of each significant feature on article shares. This will help understand the relative importance of different features in predicting article popularity.
3. **Build Predictive Models:** Develop multiple predictive models using ML/DL techniques to predict the number of shares an article.
4. **Evaluate Model Performance:** Compare the performance of different models using appropriate evaluation metrics, including R-squared, Mean Absolute Error (MAE), Mean Squared Error (MSE), and Root Mean Squared Error (RMSE).


## 1. Dataset Introduction

The Online News Popularity dataset contains metadata and social feedback (shares) for 39,797 news articles published on Mashable. Each sample has 61 attributes, including:


* **Non-predictive attributes (2):** `url` (URL of the article) and `timedelta` (days between the article publication and the dataset acquisition).
* **Predictive attributes (59):** Various features such as number of words, links, keywords, subject category, sentiment polarity, day of the week, and results from LDA topic modeling.

The target variable is `shares`, representing the number of times the article was shared on social media.

## 2. Exploratory Data Analysis (EDA)

This section details the steps taken to understand the dataset's underlying patterns, distributions, and potential anomalies.


### 2.1. Data Overview
Our dataset has been well preprocessed that there is no missing value in the dataset. We will focus on finding patterns of features. 
You can see all features from the original dataset are numeric. It should be noted that we transformed target variabe "Shares" into object variable to fulfill the classification need. There is a new feature call "followers". We will explain that in Feature Engineering part.
Here are the first 5 rows of the dateset. 

<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>n_tokens_title</th>
      <th>n_tokens_content</th>
      <th>n_unique_tokens</th>
      <th>n_non_stop_words</th>
      <th>n_non_stop_unique_tokens</th>
      <th>num_hrefs</th>
      <th>num_self_hrefs</th>
      <th>num_imgs</th>
      <th>num_videos</th>
      <th>average_token_length</th>
      <th>num_keywords</th>
      <th>data_channel_is_lifestyle</th>
      <th>data_channel_is_entertainment</th>
      <th>data_channel_is_bus</th>
      <th>data_channel_is_socmed</th>
      <th>data_channel_is_tech</th>
      <th>data_channel_is_world</th>
      <th>kw_min_min</th>
      <th>kw_max_min</th>
      <th>kw_avg_min</th>
      <th>kw_min_max</th>
      <th>kw_max_max</th>
      <th>kw_avg_max</th>
      <th>kw_min_avg</th>
      <th>kw_max_avg</th>
      <th>kw_avg_avg</th>
      <th>self_reference_min_shares</th>
      <th>self_reference_max_shares</th>
      <th>self_reference_avg_sharess</th>
      <th>weekday_is_monday</th>
      <th>weekday_is_tuesday</th>
      <th>weekday_is_wednesday</th>
      <th>weekday_is_thursday</th>
      <th>weekday_is_friday</th>
      <th>weekday_is_saturday</th>
      <th>weekday_is_sunday</th>
      <th>is_weekend</th>
      <th>LDA_00</th>
      <th>LDA_01</th>
      <th>LDA_02</th>
      <th>LDA_03</th>
      <th>LDA_04</th>
      <th>global_subjectivity</th>
      <th>global_sentiment_polarity</th>
      <th>global_rate_positive_words</th>
      <th>global_rate_negative_words</th>
      <th>rate_positive_words</th>
      <th>rate_negative_words</th>
      <th>avg_positive_polarity</th>
      <th>min_positive_polarity</th>
      <th>max_positive_polarity</th>
      <th>avg_negative_polarity</th>
      <th>min_negative_polarity</th>
      <th>max_negative_polarity</th>
      <th>title_subjectivity</th>
      <th>title_sentiment_polarity</th>
      <th>abs_title_subjectivity</th>
      <th>abs_title_sentiment_polarity</th>
      <th>shares</th>
      <th>followers</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>12</td>
      <td>219</td>
      <td>0.663594</td>
      <td>1.0</td>
      <td>0.815385</td>
      <td>4</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>4.680365</td>
      <td>5</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>496.0</td>
      <td>496.0</td>
      <td>496.000000</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.500331</td>
      <td>0.378279</td>
      <td>0.040005</td>
      <td>0.041263</td>
      <td>0.040123</td>
      <td>0.521617</td>
      <td>0.092562</td>
      <td>0.045662</td>
      <td>0.013699</td>
      <td>0.769231</td>
      <td>0.230769</td>
      <td>0.378636</td>
      <td>0.100000</td>
      <td>0.7</td>
      <td>-0.350000</td>
      <td>-0.600</td>
      <td>-0.200000</td>
      <td>0.500000</td>
      <td>-0.187500</td>
      <td>0.000000</td>
      <td>0.187500</td>
      <td>Low</td>
      <td>High</td>
    </tr>
    <tr>
      <th>1</th>
      <td>9</td>
      <td>255</td>
      <td>0.604743</td>
      <td>1.0</td>
      <td>0.791946</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>4.913725</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.799756</td>
      <td>0.050047</td>
      <td>0.050096</td>
      <td>0.050101</td>
      <td>0.050001</td>
      <td>0.341246</td>
      <td>0.148948</td>
      <td>0.043137</td>
      <td>0.015686</td>
      <td>0.733333</td>
      <td>0.266667</td>
      <td>0.286915</td>
      <td>0.033333</td>
      <td>0.7</td>
      <td>-0.118750</td>
      <td>-0.125</td>
      <td>-0.100000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.500000</td>
      <td>0.000000</td>
      <td>Low</td>
      <td>Medium</td>
    </tr>
    <tr>
      <th>2</th>
      <td>9</td>
      <td>211</td>
      <td>0.575130</td>
      <td>1.0</td>
      <td>0.663866</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>4.393365</td>
      <td>6</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>918.0</td>
      <td>918.0</td>
      <td>918.000000</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.217792</td>
      <td>0.033334</td>
      <td>0.033351</td>
      <td>0.033334</td>
      <td>0.682188</td>
      <td>0.702222</td>
      <td>0.323333</td>
      <td>0.056872</td>
      <td>0.009479</td>
      <td>0.857143</td>
      <td>0.142857</td>
      <td>0.495833</td>
      <td>0.100000</td>
      <td>1.0</td>
      <td>-0.466667</td>
      <td>-0.800</td>
      <td>-0.133333</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.500000</td>
      <td>0.000000</td>
      <td>Medium</td>
      <td>Medium</td>
    </tr>
    <tr>
      <th>3</th>
      <td>9</td>
      <td>531</td>
      <td>0.503788</td>
      <td>1.0</td>
      <td>0.665635</td>
      <td>9</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>4.404896</td>
      <td>7</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.028573</td>
      <td>0.419300</td>
      <td>0.494651</td>
      <td>0.028905</td>
      <td>0.028572</td>
      <td>0.429850</td>
      <td>0.100705</td>
      <td>0.041431</td>
      <td>0.020716</td>
      <td>0.666667</td>
      <td>0.333333</td>
      <td>0.385965</td>
      <td>0.136364</td>
      <td>0.8</td>
      <td>-0.369697</td>
      <td>-0.600</td>
      <td>-0.166667</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.500000</td>
      <td>0.000000</td>
      <td>Low</td>
      <td>Reprinted</td>
    </tr>
    <tr>
      <th>4</th>
      <td>13</td>
      <td>1072</td>
      <td>0.415646</td>
      <td>1.0</td>
      <td>0.540890</td>
      <td>19</td>
      <td>19</td>
      <td>20</td>
      <td>0</td>
      <td>4.682836</td>
      <td>7</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>545.0</td>
      <td>16000.0</td>
      <td>3151.157895</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.028633</td>
      <td>0.028794</td>
      <td>0.028575</td>
      <td>0.028572</td>
      <td>0.885427</td>
      <td>0.513502</td>
      <td>0.281003</td>
      <td>0.074627</td>
      <td>0.012127</td>
      <td>0.860215</td>
      <td>0.139785</td>
      <td>0.411127</td>
      <td>0.033333</td>
      <td>1.0</td>
      <td>-0.220192</td>
      <td>-0.500</td>
      <td>-0.050000</td>
      <td>0.454545</td>
      <td>0.136364</td>
      <td>0.045455</td>
      <td>0.136364</td>
      <td>Low</td>
      <td>Low</td>
    </tr>
  </tbody>
</table>
</div>



### 2.2. Target Variable Analysis

Here we divided articles into four groups based on their shares. 

#### Shares Distribution
| Category | Count | Percentage(%) |
|----------|-------|--------------|
| Low | 18,490 | 46.64 |
| Medium | 13,075 | 32.98 |
| High | 6,079 | 15.33 |
| Viral | 2,000 | 5.04 |

We also calculated their proportion.
#### Followers Distribution
| Category | Count |
|----------|-------|
| Low | 15,234 |
| Unknown | 9,297 |
| Medium | 8,200 |
| Reprinted | 3,747 |
| Extremely Low | 2,163 |

We plot pie chart and histogram to show the distribution intuitively.

![png](Final%20notebook%20images/Article%20Shares%20Category%20Distribution%20Pie%20Chart.png)

![png](Final%20notebook%20images/Article%20Shares%20Category%20Distribution%20Histogram.png)
Most of the articles do not really have a high number of views, which contributes to the long-tailed character of this data. Such an imbalanced dataset is the biggest factor that affects the performance of the model in this study.


### 2.3. Content Feature Analysis
One of the major characteristics of our selected dataset is that its features comprehensively and detailedly summarize the content of the articles. This includes word count statistics, sentiment analysis, and keyword analysis, etc. In this section we will focus on analyzing word count statistics.

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
      <th>cv</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>n_tokens_title</th>
      <td>39644.0</td>
      <td>10.398749</td>
      <td>2.114037</td>
      <td>2.0</td>
      <td>9.000000</td>
      <td>10.000000</td>
      <td>12.000000</td>
      <td>23.000000</td>
      <td>0.203297</td>
    </tr>
    <tr>
      <th>n_tokens_content</th>
      <td>39644.0</td>
      <td>546.514731</td>
      <td>471.107508</td>
      <td>0.0</td>
      <td>246.000000</td>
      <td>409.000000</td>
      <td>716.000000</td>
      <td>8474.000000</td>
      <td>0.862022</td>
    </tr>
    <tr>
      <th>n_unique_tokens</th>
      <td>39644.0</td>
      <td>0.548216</td>
      <td>3.520708</td>
      <td>0.0</td>
      <td>0.470870</td>
      <td>0.539226</td>
      <td>0.608696</td>
      <td>701.000000</td>
      <td>6.422122</td>
    </tr>
    <tr>
      <th>num_hrefs</th>
      <td>39644.0</td>
      <td>10.883690</td>
      <td>11.332017</td>
      <td>0.0</td>
      <td>4.000000</td>
      <td>8.000000</td>
      <td>14.000000</td>
      <td>304.000000</td>
      <td>1.041193</td>
    </tr>
    <tr>
      <th>num_self_hrefs</th>
      <td>39644.0</td>
      <td>3.293638</td>
      <td>3.855141</td>
      <td>0.0</td>
      <td>1.000000</td>
      <td>3.000000</td>
      <td>4.000000</td>
      <td>116.000000</td>
      <td>1.170481</td>
    </tr>
    <tr>
      <th>num_imgs</th>
      <td>39644.0</td>
      <td>4.544143</td>
      <td>8.309434</td>
      <td>0.0</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>4.000000</td>
      <td>128.000000</td>
      <td>1.828603</td>
    </tr>
    <tr>
      <th>num_videos</th>
      <td>39644.0</td>
      <td>1.249874</td>
      <td>4.107855</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>91.000000</td>
      <td>3.286616</td>
    </tr>
    <tr>
      <th>average_token_length</th>
      <td>39644.0</td>
      <td>4.548239</td>
      <td>0.844406</td>
      <td>0.0</td>
      <td>4.478404</td>
      <td>4.664082</td>
      <td>4.854839</td>
      <td>8.041534</td>
      <td>0.185655</td>
    </tr>
    <tr>
      <th>num_keywords</th>
      <td>39644.0</td>
      <td>7.223767</td>
      <td>1.909130</td>
      <td>1.0</td>
      <td>6.000000</td>
      <td>7.000000</td>
      <td>9.000000</td>
      <td>10.000000</td>
      <td>0.264285</td>
    </tr>
  </tbody>
</table>
</div>


We also plot the boxplots of the nine selected features.
![png](Final%20notebook%20images/Content%20feature%20statistics.png)
We found that many of the features are significantly right-skewed and appear to have a lot of positive outliers. and except for the differences in titles and keywords, the remaining features are clustered in a very small interval. This suggests that most of the articles have similar vocabulary size and vocabulary type, which may be some kind of characteristic or consensus of news industry articles. 

### 2.4. Channel Distribution Analysis
The dataset provides categories for each article, such as entertainment and technology news. Here is the distribution of channels
![png](Final%20notebook%20images/Article%20Channel%20Distribution.png)
It can be noticed that the website focused on WORLD, TECHNOLOGY, ENTERTAINMENT and BUSINESS news.


### 2.5. Followers Analysis



### 2.6. Selected Features Analysis
We select the following features we believe are important for predicting the number of shares. We draw boxplots for these features.
![png](EDA%20and%20Feature%20Engineering_files/EDA%20and%20Feature%20Engineering_13_0.png)
Among them, n_tokens_content shows the greatest variation and has many outliers. In contrast, num_hrefs, num_imgs, and num_videos are relatively stable.

We also use IQR and Z-score to detect outliers in n_tokens_content.
```
    Number of outliers using IQR: 4541
    Number of outliers using Z-score: 308
```


### 2.6. Feature Correlation Analysis
We compute the correlation between the target variable and the selected features. The log_shares and boxcox_shares are also considered.

Highest correlated features:
```
    shares                        1.000000
    log_shares                    0.510181
    boxcox_shares                 0.416169
    kw_avg_avg                    0.110413
    LDA_03                        0.083771
    kw_max_avg                    0.064306
    self_reference_avg_sharess    0.057789
    self_reference_min_shares     0.055958
    self_reference_max_shares     0.047115
    num_hrefs                     0.045404
    kw_avg_max                    0.044686
    kw_min_avg                    0.039551
    num_imgs                      0.039388
    global_subjectivity           0.031604
    kw_avg_min                    0.030406
    Name: shares, dtype: float64
```

Lowest correlated features:
```
    rate_negative_words             -0.005183
    weekday_is_tuesday              -0.007941
    weekday_is_thursday             -0.008833
    LDA_01                          -0.010183
    data_channel_is_bus             -0.012376
    rate_positive_words             -0.013241
    data_channel_is_tech            -0.013253
    LDA_04                          -0.016622
    data_channel_is_entertainment   -0.017006
    min_negative_polarity           -0.019297
    max_negative_polarity           -0.019300
    average_token_length            -0.022007
    avg_negative_polarity           -0.032029
    data_channel_is_world           -0.049497
    LDA_02                          -0.059163
    Name: shares, dtype: float64
```

Based on our analysis, the following variables are likely to be important in predicting shares: log_shares, boxcox_shares, kw_avg_avg, LDA_03, kw_max_avg, self_reference_avg_sharess. Additionally, num_hrefs (number of hyperlinks) and num_imgs (number of images) also show a certain level of correlation and may have some value for modeling.


### 2.7. High vs. Low Shares Comparison with Hyperlinks

Next, we compare the number of hyperlinks (num_hrefs) between High Shares and Low Shares to analyze whether the number of hyperlinks affects the number of article shares.
![png](EDA%20and%20Feature%20Engineering_files/EDA%20and%20Feature%20Engineering_17_0.png)
The analysis results show that the median and IQR of both groups (High Shares & Low Shares) are similar. Additionally, there are many outliers, and the distribution of num_hrefs in high-share and low-share articles is nearly identical.

Therefore, we conclude that the number of hyperlinks is likely not a major factor influencing the number of article shares.


### 2.8. Shares Category Analysis
To better analyze how different number of shares interact with other features, we create a new categorical variable shares_category based on the quantiles of the shares. 4 categories are created: Low, Medium-Low, Medium-High, High. We can notice that the number of articles in each category is quite balanced.
```
    shares_category
    Medium-Low     10152
    Medium-High     9932
    Low             9930
    High            9630
    Name: count, dtype: int64
```

We draw two charts. The first chart displays the mean values of different features across various share categories (Low, Medium-Low, Medium-High, High). 
![png](EDA%20and%20Feature%20Engineering_files/EDA%20and%20Feature%20Engineering_19_1.png)

Since log_shares has the highest correlation with shares, it helps reduce data skewness. kw_avg_avg and kw_max_avg show relatively high correlations with shares, indicating that keyword weight might significantly influence shares. num_hrefs and num_imgs were identified as potentially influential variables during the EDA process. Although the previous step suggested that their impact might be limited, they still hold some value and need further verification.

From the analysis, we observe that log_shares increases as shares_category rises, confirming that log_shares is an important transformation of shares. kw_avg_avg and kw_max_avg show a relatively stable increasing trend across different categories, suggesting that keyword influence is likely associated with the number of shares. In contrast, num_hrefs and num_imgs have smaller values and show no significant trend changes, indicating a weaker impact.

The second chart presents the mean log_shares across different share categories.
![png](EDA%20and%20Feature%20Engineering_files/EDA%20and%20Feature%20Engineering_19_2.png)
We can conclude that log_shares gradually increases with shares_category, demonstrating that the logarithmic transformation effectively captures the distribution trend of shares. Additionally, log_shares in high shares_category is significantly higher than in low shares_category, further proving the effectiveness of log_shares as a key variable.


## 3. Feature Engineering
There is no missing value in the dataset. So we will focus on handling the skewness of the features.

### 3.1. Data Conversion

We convert the binary categorical features to boolean type.
Here is the list of the 14 selected features:
```
bool_features = [
    "weekday_is_monday",
    "weekday_is_tuesday",
    "weekday_is_wednesday",
    "weekday_is_thursday",
    "weekday_is_friday",
    "weekday_is_saturday",
    "weekday_is_sunday",
    "is_weekend",
    "data_channel_is_lifestyle",
    "data_channel_is_entertainment",
    "data_channel_is_bus",
    "data_channel_is_socmed",
    "data_channel_is_tech",
    "data_channel_is_world"
]
```

### 3.2. Skewness Handling
First, we plot the histogram of the all features.
![png](EDA%20and%20Feature%20Engineering_files/EDA%20and%20Feature%20Engineering_26_0.png)

![png](EDA%20and%20Feature%20Engineering_files/EDA%20and%20Feature%20Engineering_26_1.png)

![png](EDA%20and%20Feature%20Engineering_files/EDA%20and%20Feature%20Engineering_26_2.png)

![png](EDA%20and%20Feature%20Engineering_files/EDA%20and%20Feature%20Engineering_26_3.png)

![png](EDA%20and%20Feature%20Engineering_files/EDA%20and%20Feature%20Engineering_26_4.png)

![png](EDA%20and%20Feature%20Engineering_files/EDA%20and%20Feature%20Engineering_26_5.png)

![png](EDA%20and%20Feature%20Engineering_files/EDA%20and%20Feature%20Engineering_26_6.png)

It can be seen that there are many severely skewed features. We calculate the skewness of each feature.

```
    n_tokens_title: 0.16532037674928027
    n_tokens_content: 2.9454219387867084
    n_unique_tokens: 198.65511559825592
    n_non_stop_words: 198.7924453768874
    n_non_stop_unique_tokens: 198.44329440926512
    num_hrefs: 4.013494828201318
    num_self_hrefs: 5.172751105757634
    num_imgs: 3.9465958446535474
    num_videos: 7.0195327862958665
    average_token_length: -4.57601155020474
    num_keywords: -0.14725125199950523
    data_channel_is_lifestyle: 3.9930191433554167
    data_channel_is_entertainment: 1.68358480940472
    data_channel_is_bus: 1.8768701859879158
    data_channel_is_socmed: 3.758879630973088
    data_channel_is_tech: 1.6199757646890423
    data_channel_is_world: 1.4051693841208097
    kw_min_min: 2.3749472801825444
    kw_max_min: 35.32843373115432
    kw_avg_min: 31.306108102660584
    kw_min_max: 10.386371634782769
    kw_max_max: -2.6449817621966782
    kw_avg_max: 0.6243096463608944
    kw_min_avg: 0.4679758464905322
    kw_max_avg: 16.41166955537124
    kw_avg_avg: 5.760177291618559
    self_reference_min_shares: 26.264364160300094
    self_reference_max_shares: 13.870849049433598
    self_reference_avg_sharess: 17.9140933776756
    weekday_is_monday: 1.7759082442285052
    weekday_is_tuesday: 1.6105470619092879
    weekday_is_wednesday: 1.6009709768881089
    weekday_is_thursday: 1.6370700482983118
    weekday_is_friday: 2.030304835180609
    weekday_is_saturday: 3.6370857599701125
    weekday_is_sunday: 3.3999273763003046
    is_weekend: 2.18850033431371
    LDA_00: 1.5674632332004765
    LDA_01: 2.0867218234169407
    LDA_02: 1.311694902028395
    LDA_03: 1.2387159863782728
    LDA_04: 1.1731294759766238
    global_subjectivity: -1.3726888305603973
    global_sentiment_polarity: 0.10545709665820545
    global_rate_positive_words: 0.32304661115048916
    global_rate_negative_words: 1.491917309190822
    rate_positive_words: -1.423105853002299
    rate_negative_words: 0.4072406539941212
    avg_positive_polarity: -0.7247949503201233
    min_positive_polarity: 3.0404677374643283
    max_positive_polarity: -0.9397564591253907
    avg_negative_polarity: -0.55164402900095
    min_negative_polarity: -0.07315481617331099
    max_negative_polarity: -3.4597470578480207
    title_subjectivity: 0.816084749635643
    title_sentiment_polarity: 0.39610883665169594
    abs_title_subjectivity: -0.6241493828840421
    abs_title_sentiment_polarity: 1.7041934399140888
    log_shares: 1.0264770065011177
    boxcox_shares: -0.22394970896715033
    data_channel: N/A
    zscore: 33.96388487571418
    high_low_category: N/A
```

To handle the skewness, we classify the features into five categories and perform different transformations methods. This is based on the skewness value. Note that the boolean features are not included in the transformation.
- **Skewness > 10:** These features are highly right-skewed. We apply Yeo-Johnson transformation to them. The Yeo-Johnson transformation is a power transformation that can handle both positive and negative skewness.
- **Skewness < -10:** These features are highly left-skewed. We apply inverse transformation to them. The inverse transformation is defined as `1/x`.
- **2 < Skewness < 10:** These features are moderately right-skewed. We apply square transformation to them. The square transformation is defined as `x^2`. It can reduce the skewness and make the distribution more normal.
- **-10 < Skewness < -2:** These features are moderately left-skewed. We first take the negative value of the features, then apply square root transformation to them. After that, take the negative value of the result to restore the original sign.
- **-2 < Skewness < 2:** These features are not significantly skewed. We do not need to apply any transformation to them.

Here is the skewness of the features after the transformation. We can see that the skewness of most features are indeed reduced.
```
    n_tokens_title: 0.16532037674928027
    n_tokens_content: 2.9454219387867084
    n_unique_tokens: 198.65511559825592
    n_non_stop_words: 198.7924453768874
    n_non_stop_unique_tokens: 198.44329440926512
    num_hrefs: 4.013494828201318
    num_self_hrefs: 5.172751105757634
    num_imgs: 3.9465958446535474
    num_videos: 7.0195327862958665
    average_token_length: -4.57601155020474
    num_keywords: -0.14725125199950523
    data_channel_is_lifestyle: 3.9930191433554167
    data_channel_is_entertainment: 1.68358480940472
    data_channel_is_bus: 1.8768701859879158
    data_channel_is_socmed: 3.758879630973088
    data_channel_is_tech: 1.6199757646890423
    data_channel_is_world: 1.4051693841208097
    kw_min_min: 2.3749472801825444
    kw_max_min: 35.32843373115432
    kw_avg_min: 31.306108102660584
    kw_min_max: 10.386371634782769
    kw_max_max: -2.6449817621966782
    kw_avg_max: 0.6243096463608944
    kw_min_avg: 0.4679758464905322
    kw_max_avg: 16.41166955537124
    kw_avg_avg: 5.760177291618559
    self_reference_min_shares: 26.264364160300094
    self_reference_max_shares: 13.870849049433598
    self_reference_avg_sharess: 17.9140933776756
    weekday_is_monday: 1.7759082442285052
    weekday_is_tuesday: 1.6105470619092879
    weekday_is_wednesday: 1.6009709768881089
    weekday_is_thursday: 1.6370700482983118
    weekday_is_friday: 2.030304835180609
    weekday_is_saturday: 3.6370857599701125
    weekday_is_sunday: 3.3999273763003046
    is_weekend: 2.18850033431371
    LDA_00: 1.5674632332004765
    LDA_01: 2.0867218234169407
    LDA_02: 1.311694902028395
    LDA_03: 1.2387159863782728
    LDA_04: 1.1731294759766238
    global_subjectivity: -1.3726888305603973
    global_sentiment_polarity: 0.10545709665820545
    global_rate_positive_words: 0.32304661115048916
    global_rate_negative_words: 1.491917309190822
    rate_positive_words: -1.423105853002299
    rate_negative_words: 0.4072406539941212
    avg_positive_polarity: -0.7247949503201233
    min_positive_polarity: 3.0404677374643283
    max_positive_polarity: -0.9397564591253907
    avg_negative_polarity: -0.55164402900095
    min_negative_polarity: -0.07315481617331099
    max_negative_polarity: -3.4597470578480207
    title_subjectivity: 0.816084749635643
    title_sentiment_polarity: 0.39610883665169594
    abs_title_subjectivity: -0.6241493828840421
    abs_title_sentiment_polarity: 1.7041934399140888
    log_shares: 1.0264770065011177
    boxcox_shares: -0.22394970896715033
    data_channel: N/A
    zscore: 33.96388487571418
    high_low_category: N/A    
```

## 4. Model Building (not yet finished)

## 5. Model Evaluation (not yet finished)

## 6. Conclusion (not yet finished)
