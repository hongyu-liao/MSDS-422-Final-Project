```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats
from scipy.stats import boxcox
```


```python
# Load data
file_path = "Data/OnlineNewsPopularity.csv"
df = pd.read_csv(file_path)
df.columns = df.columns.str.strip()
```


```python
# Display dataset information
print("Dataset Information:")
df.info()
print("\nFirst 5 rows:")
print(df.head())
```

    Dataset Information:
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 39644 entries, 0 to 39643
    Data columns (total 61 columns):
     #   Column                         Non-Null Count  Dtype  
    ---  ------                         --------------  -----  
     0   url                            39644 non-null  object 
     1   timedelta                      39644 non-null  float64
     2   n_tokens_title                 39644 non-null  float64
     3   n_tokens_content               39644 non-null  float64
     4   n_unique_tokens                39644 non-null  float64
     5   n_non_stop_words               39644 non-null  float64
     6   n_non_stop_unique_tokens       39644 non-null  float64
     7   num_hrefs                      39644 non-null  float64
     8   num_self_hrefs                 39644 non-null  float64
     9   num_imgs                       39644 non-null  float64
     10  num_videos                     39644 non-null  float64
     11  average_token_length           39644 non-null  float64
     12  num_keywords                   39644 non-null  float64
     13  data_channel_is_lifestyle      39644 non-null  float64
     14  data_channel_is_entertainment  39644 non-null  float64
     15  data_channel_is_bus            39644 non-null  float64
     16  data_channel_is_socmed         39644 non-null  float64
     17  data_channel_is_tech           39644 non-null  float64
     18  data_channel_is_world          39644 non-null  float64
     19  kw_min_min                     39644 non-null  float64
     20  kw_max_min                     39644 non-null  float64
     21  kw_avg_min                     39644 non-null  float64
     22  kw_min_max                     39644 non-null  float64
     23  kw_max_max                     39644 non-null  float64
     24  kw_avg_max                     39644 non-null  float64
     25  kw_min_avg                     39644 non-null  float64
     26  kw_max_avg                     39644 non-null  float64
     27  kw_avg_avg                     39644 non-null  float64
     28  self_reference_min_shares      39644 non-null  float64
     29  self_reference_max_shares      39644 non-null  float64
     30  self_reference_avg_sharess     39644 non-null  float64
     31  weekday_is_monday              39644 non-null  float64
     32  weekday_is_tuesday             39644 non-null  float64
     33  weekday_is_wednesday           39644 non-null  float64
     34  weekday_is_thursday            39644 non-null  float64
     35  weekday_is_friday              39644 non-null  float64
     36  weekday_is_saturday            39644 non-null  float64
     37  weekday_is_sunday              39644 non-null  float64
     38  is_weekend                     39644 non-null  float64
     39  LDA_00                         39644 non-null  float64
     40  LDA_01                         39644 non-null  float64
     41  LDA_02                         39644 non-null  float64
     42  LDA_03                         39644 non-null  float64
     43  LDA_04                         39644 non-null  float64
     44  global_subjectivity            39644 non-null  float64
     45  global_sentiment_polarity      39644 non-null  float64
     46  global_rate_positive_words     39644 non-null  float64
     47  global_rate_negative_words     39644 non-null  float64
     48  rate_positive_words            39644 non-null  float64
     49  rate_negative_words            39644 non-null  float64
     50  avg_positive_polarity          39644 non-null  float64
     51  min_positive_polarity          39644 non-null  float64
     52  max_positive_polarity          39644 non-null  float64
     53  avg_negative_polarity          39644 non-null  float64
     54  min_negative_polarity          39644 non-null  float64
     55  max_negative_polarity          39644 non-null  float64
     56  title_subjectivity             39644 non-null  float64
     57  title_sentiment_polarity       39644 non-null  float64
     58  abs_title_subjectivity         39644 non-null  float64
     59  abs_title_sentiment_polarity   39644 non-null  float64
     60  shares                         39644 non-null  int64  
    dtypes: float64(59), int64(1), object(1)
    memory usage: 18.5+ MB
    
    First 5 rows:
                                                     url  timedelta  \
    0  http://mashable.com/2013/01/07/amazon-instant-...      731.0   
    1  http://mashable.com/2013/01/07/ap-samsung-spon...      731.0   
    2  http://mashable.com/2013/01/07/apple-40-billio...      731.0   
    3  http://mashable.com/2013/01/07/astronaut-notre...      731.0   
    4   http://mashable.com/2013/01/07/att-u-verse-apps/      731.0   
    
       n_tokens_title  n_tokens_content  n_unique_tokens  n_non_stop_words  \
    0            12.0             219.0         0.663594               1.0   
    1             9.0             255.0         0.604743               1.0   
    2             9.0             211.0         0.575130               1.0   
    3             9.0             531.0         0.503788               1.0   
    4            13.0            1072.0         0.415646               1.0   
    
       n_non_stop_unique_tokens  num_hrefs  num_self_hrefs  num_imgs  ...  \
    0                  0.815385        4.0             2.0       1.0  ...   
    1                  0.791946        3.0             1.0       1.0  ...   
    2                  0.663866        3.0             1.0       1.0  ...   
    3                  0.665635        9.0             0.0       1.0  ...   
    4                  0.540890       19.0            19.0      20.0  ...   
    
       min_positive_polarity  max_positive_polarity  avg_negative_polarity  \
    0               0.100000                    0.7              -0.350000   
    1               0.033333                    0.7              -0.118750   
    2               0.100000                    1.0              -0.466667   
    3               0.136364                    0.8              -0.369697   
    4               0.033333                    1.0              -0.220192   
    
       min_negative_polarity  max_negative_polarity  title_subjectivity  \
    0                 -0.600              -0.200000            0.500000   
    1                 -0.125              -0.100000            0.000000   
    2                 -0.800              -0.133333            0.000000   
    3                 -0.600              -0.166667            0.000000   
    4                 -0.500              -0.050000            0.454545   
    
       title_sentiment_polarity  abs_title_subjectivity  \
    0                 -0.187500                0.000000   
    1                  0.000000                0.500000   
    2                  0.000000                0.500000   
    3                  0.000000                0.500000   
    4                  0.136364                0.045455   
    
       abs_title_sentiment_polarity  shares  
    0                      0.187500     593  
    1                      0.000000     711  
    2                      0.000000    1500  
    3                      0.000000    1200  
    4                      0.136364     505  
    
    [5 rows x 61 columns]
    


```python
# Statistical summary
print("\nStatistical Summary:")
print(df.describe())
```

    
    Statistical Summary:
              timedelta  n_tokens_title  n_tokens_content  n_unique_tokens  \
    count  39644.000000    39644.000000      39644.000000     39644.000000   
    mean     354.530471       10.398749        546.514731         0.548216   
    std      214.163767        2.114037        471.107508         3.520708   
    min        8.000000        2.000000          0.000000         0.000000   
    25%      164.000000        9.000000        246.000000         0.470870   
    50%      339.000000       10.000000        409.000000         0.539226   
    75%      542.000000       12.000000        716.000000         0.608696   
    max      731.000000       23.000000       8474.000000       701.000000   
    
           n_non_stop_words  n_non_stop_unique_tokens     num_hrefs  \
    count      39644.000000              39644.000000  39644.000000   
    mean           0.996469                  0.689175     10.883690   
    std            5.231231                  3.264816     11.332017   
    min            0.000000                  0.000000      0.000000   
    25%            1.000000                  0.625739      4.000000   
    50%            1.000000                  0.690476      8.000000   
    75%            1.000000                  0.754630     14.000000   
    max         1042.000000                650.000000    304.000000   
    
           num_self_hrefs      num_imgs    num_videos  ...  min_positive_polarity  \
    count    39644.000000  39644.000000  39644.000000  ...           39644.000000   
    mean         3.293638      4.544143      1.249874  ...               0.095446   
    std          3.855141      8.309434      4.107855  ...               0.071315   
    min          0.000000      0.000000      0.000000  ...               0.000000   
    25%          1.000000      1.000000      0.000000  ...               0.050000   
    50%          3.000000      1.000000      0.000000  ...               0.100000   
    75%          4.000000      4.000000      1.000000  ...               0.100000   
    max        116.000000    128.000000     91.000000  ...               1.000000   
    
           max_positive_polarity  avg_negative_polarity  min_negative_polarity  \
    count           39644.000000           39644.000000           39644.000000   
    mean                0.756728              -0.259524              -0.521944   
    std                 0.247786               0.127726               0.290290   
    min                 0.000000              -1.000000              -1.000000   
    25%                 0.600000              -0.328383              -0.700000   
    50%                 0.800000              -0.253333              -0.500000   
    75%                 1.000000              -0.186905              -0.300000   
    max                 1.000000               0.000000               0.000000   
    
           max_negative_polarity  title_subjectivity  title_sentiment_polarity  \
    count           39644.000000        39644.000000              39644.000000   
    mean               -0.107500            0.282353                  0.071425   
    std                 0.095373            0.324247                  0.265450   
    min                -1.000000            0.000000                 -1.000000   
    25%                -0.125000            0.000000                  0.000000   
    50%                -0.100000            0.150000                  0.000000   
    75%                -0.050000            0.500000                  0.150000   
    max                 0.000000            1.000000                  1.000000   
    
           abs_title_subjectivity  abs_title_sentiment_polarity         shares  
    count            39644.000000                  39644.000000   39644.000000  
    mean                 0.341843                      0.156064    3395.380184  
    std                  0.188791                      0.226294   11626.950749  
    min                  0.000000                      0.000000       1.000000  
    25%                  0.166667                      0.000000     946.000000  
    50%                  0.500000                      0.000000    1400.000000  
    75%                  0.500000                      0.250000    2800.000000  
    max                  0.500000                      1.000000  843300.000000  
    
    [8 rows x 60 columns]
    


```python
# Check for missing values
missing_values = df.isnull().sum()
missing_values = missing_values[missing_values > 0]
if not missing_values.empty:
    print("\nMissing Values:")
    print(missing_values)
else:
    print("\nNo missing values found in the dataset.")
```

    
    No missing values found in the dataset.
    


```python
# Visualize the distribution of the target variable 'shares'
plt.figure(figsize=(12, 6))
sns.histplot(df['shares'], bins=50, kde=True)
plt.title("Distribution of Shares")
plt.xlabel("Shares")
plt.ylabel("Frequency")
plt.show()

```


    
![png](EDA%20and%20Feature%20Engineering_files/EDA%20and%20Feature%20Engineering_5_0.png)
    



```python
# Log-transform the 'shares' column to handle skewness
df['log_shares'] = np.log1p(df['shares'])
plt.figure(figsize=(12, 6))
sns.histplot(df['log_shares'], bins=50, kde=True)
plt.title("Log-transformed Distribution of Shares")
plt.xlabel("Log Shares")
plt.ylabel("Frequency")
plt.show()
```


    
![png](EDA%20and%20Feature%20Engineering_files/EDA%20and%20Feature%20Engineering_6_0.png)
    



```python
# Apply Box-Cox transformation to further normalize the distribution
df['boxcox_shares'], lambda_val = boxcox(df['shares'] + 1)  # Adding 1 to handle zero values
print(f"Optimal lambda for Box-Cox: {lambda_val}")
plt.figure(figsize=(12, 6))
sns.histplot(df['boxcox_shares'], bins=50, kde=True)
plt.title("Box-Cox Transformed Distribution of Shares")
plt.xlabel("Box-Cox Shares")
plt.ylabel("Frequency")
plt.show()
```

    Optimal lambda for Box-Cox: -0.21964795585426122
    


    
![png](EDA%20and%20Feature%20Engineering_files/EDA%20and%20Feature%20Engineering_7_1.png)
    



```python
# Compute and visualize correlation matrix using numerical columns
numeric_df = df.select_dtypes(include=['number'])
corr_matrix = numeric_df.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, cmap="coolwarm", annot=False, linewidths=0.5)
plt.title("Feature Correlation Matrix")
plt.show()
```


    
![png](EDA%20and%20Feature%20Engineering_files/EDA%20and%20Feature%20Engineering_8_0.png)
    



```python
# Identify the most correlated feature pairs
high_corr_pairs = corr_matrix.unstack().sort_values(ascending=False)
print("Top correlated feature pairs:")
print(high_corr_pairs[high_corr_pairs < 1].head(10))

```

    Top correlated feature pairs:
    n_non_stop_unique_tokens  n_unique_tokens             0.999852
    n_unique_tokens           n_non_stop_unique_tokens    0.999852
    n_non_stop_words          n_unique_tokens             0.999572
    n_unique_tokens           n_non_stop_words            0.999572
    n_non_stop_unique_tokens  n_non_stop_words            0.999532
    n_non_stop_words          n_non_stop_unique_tokens    0.999532
    log_shares                boxcox_shares               0.981244
    boxcox_shares             log_shares                  0.981244
    kw_max_min                kw_avg_min                  0.940529
    kw_avg_min                kw_max_min                  0.940529
    dtype: float64
    


```python
# Analyze effect of weekdays on shares
plt.figure(figsize=(10, 5))
sns.boxplot(x='weekday_is_monday', y='shares', data=df)
plt.title("Effect of Monday Publication on Shares")
plt.show()
```


    
![png](EDA%20and%20Feature%20Engineering_files/EDA%20and%20Feature%20Engineering_10_0.png)
    



```python
# Effect of data channel on shares
channels = ['data_channel_is_lifestyle', 'data_channel_is_entertainment', 
            'data_channel_is_bus', 'data_channel_is_socmed', 
            'data_channel_is_tech', 'data_channel_is_world']

df['data_channel'] = df[channels].idxmax(axis=1)
df['data_channel'] = df['data_channel'].str.replace('data_channel_is_', '')

plt.figure(figsize=(10, 6))
sns.barplot(x=df['data_channel'], y=df['shares'], estimator=np.mean, errorbar=None)
plt.title("Average Shares by Data Channel")
plt.xlabel("Data Channel")
plt.ylabel("Average Shares")
plt.xticks(rotation=45)
plt.show()
```


    
![png](EDA%20and%20Feature%20Engineering_files/EDA%20and%20Feature%20Engineering_11_0.png)
    



```python
# Scatterplot to visualize the relationship between article length and shares
plt.figure(figsize=(10, 6))
sns.scatterplot(x='n_tokens_content', y='shares', data=df)
plt.title("Article Length vs. Shares")
plt.show()

```


    
![png](EDA%20and%20Feature%20Engineering_files/EDA%20and%20Feature%20Engineering_12_0.png)
    



```python
# Boxplot for selected numerical features
numerical_features = ['n_tokens_content', 'num_hrefs', 'num_imgs', 'num_videos']
plt.figure(figsize=(12, 8))
df[numerical_features].boxplot()
plt.xticks(rotation=45)
plt.title("Boxplot of Selected Numerical Features")
plt.show()
```


    
![png](EDA%20and%20Feature%20Engineering_files/EDA%20and%20Feature%20Engineering_13_0.png)
    



```python
# Detect outliers using the IQR method
Q1 = df['shares'].quantile(0.25)
Q3 = df['shares'].quantile(0.75)
IQR = Q3 - Q1
outliers = df[(df['shares'] < (Q1 - 1.5 * IQR)) | (df['shares'] > (Q3 + 1.5 * IQR))]
print(f"Number of outliers using IQR: {len(outliers)}")


```

    Number of outliers using IQR: 4541
    


```python
# Detect outliers using Z-score
df['zscore'] = stats.zscore(df['shares'])
outliers_zscore = df[df['zscore'].abs() > 3]
print(f"Number of outliers using Z-score: {len(outliers_zscore)}")
```

    Number of outliers using Z-score: 308
    


```python
# Compute correlation between all variables and 'shares'
corr_with_shares = numeric_df.corr()['shares'].sort_values(ascending=False)
print(corr_with_shares.head(15))
print(corr_with_shares.tail(15))
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
    


```python
# High vs. Low shares comparison
threshold_high = df['shares'].quantile(0.75)
threshold_low = df['shares'].quantile(0.25)

df['high_low_category'] = np.where(df['shares'] > threshold_high, 'High Shares', 'Low Shares')

plt.figure(figsize=(12, 6))
sns.boxplot(x='high_low_category', y='num_hrefs', data=df)
plt.title("Comparison of Number of Hrefs for High vs. Low Shares")
plt.show()
```


    
![png](EDA%20and%20Feature%20Engineering_files/EDA%20and%20Feature%20Engineering_17_0.png)
    



```python
# Ensure 'shares_category' column exists for grouping
q1 = df['shares'].quantile(0.25)  # First quartile
q2 = df['shares'].quantile(0.5)   # Median
q3 = df['shares'].quantile(0.75)  # Third quartile

# Create categorical variable for shares
df['shares_category'] = pd.cut(df['shares'], 
                               bins=[0, q1, q2, q3, df['shares'].max()], 
                               labels=['Low', 'Medium-Low', 'Medium-High', 'High'])

# Verify creation of 'shares_category'
print(df['shares_category'].value_counts())  # Check distribution of categories
```

    shares_category
    Medium-Low     10152
    Medium-High     9932
    Low             9930
    High            9630
    Name: count, dtype: int64
    


```python
# Visualizing feature differences across share categories using subplots
fig, axes = plt.subplots(2, 1, figsize=(12, 10))

# Select features based on correlation analysis
selected_features = ['log_shares', 'kw_avg_avg', 'num_hrefs', 'num_imgs', 'kw_max_avg']

# Plot bar chart for selected features across share categories
df.groupby('shares_category')[selected_features].mean().plot(kind='bar', ax=axes[0])
axes[0].set_title("Feature Differences Across Share Categories (Updated Features)")
axes[0].tick_params(axis='x', rotation=45)

# Plot mean log_shares across categories separately
df.groupby('shares_category')['log_shares'].mean().plot(kind='bar', ax=axes[1], color='green')
axes[1].set_title("Log Shares Across Share Categories")
axes[1].tick_params(axis='x', rotation=45)

# Improve layout and display
plt.tight_layout()
plt.show()
```

    C:\Users\Hongyu\AppData\Local\Temp\ipykernel_4696\40519870.py:8: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      df.groupby('shares_category')[selected_features].mean().plot(kind='bar', ax=axes[0])
    C:\Users\Hongyu\AppData\Local\Temp\ipykernel_4696\40519870.py:13: FutureWarning: The default of observed=False is deprecated and will be changed to True in a future version of pandas. Pass observed=False to retain current behavior or observed=True to adopt the future default and silence this warning.
      df.groupby('shares_category')['log_shares'].mean().plot(kind='bar', ax=axes[1], color='green')
    


    
![png](EDA%20and%20Feature%20Engineering_files/EDA%20and%20Feature%20Engineering_19_1.png)
    


fe


```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 39644 entries, 0 to 39643
    Data columns (total 67 columns):
     #   Column                         Non-Null Count  Dtype   
    ---  ------                         --------------  -----   
     0   url                            39644 non-null  object  
     1   timedelta                      39644 non-null  float64 
     2   n_tokens_title                 39644 non-null  float64 
     3   n_tokens_content               39644 non-null  float64 
     4   n_unique_tokens                39644 non-null  float64 
     5   n_non_stop_words               39644 non-null  float64 
     6   n_non_stop_unique_tokens       39644 non-null  float64 
     7   num_hrefs                      39644 non-null  float64 
     8   num_self_hrefs                 39644 non-null  float64 
     9   num_imgs                       39644 non-null  float64 
     10  num_videos                     39644 non-null  float64 
     11  average_token_length           39644 non-null  float64 
     12  num_keywords                   39644 non-null  float64 
     13  data_channel_is_lifestyle      39644 non-null  float64 
     14  data_channel_is_entertainment  39644 non-null  float64 
     15  data_channel_is_bus            39644 non-null  float64 
     16  data_channel_is_socmed         39644 non-null  float64 
     17  data_channel_is_tech           39644 non-null  float64 
     18  data_channel_is_world          39644 non-null  float64 
     19  kw_min_min                     39644 non-null  float64 
     20  kw_max_min                     39644 non-null  float64 
     21  kw_avg_min                     39644 non-null  float64 
     22  kw_min_max                     39644 non-null  float64 
     23  kw_max_max                     39644 non-null  float64 
     24  kw_avg_max                     39644 non-null  float64 
     25  kw_min_avg                     39644 non-null  float64 
     26  kw_max_avg                     39644 non-null  float64 
     27  kw_avg_avg                     39644 non-null  float64 
     28  self_reference_min_shares      39644 non-null  float64 
     29  self_reference_max_shares      39644 non-null  float64 
     30  self_reference_avg_sharess     39644 non-null  float64 
     31  weekday_is_monday              39644 non-null  float64 
     32  weekday_is_tuesday             39644 non-null  float64 
     33  weekday_is_wednesday           39644 non-null  float64 
     34  weekday_is_thursday            39644 non-null  float64 
     35  weekday_is_friday              39644 non-null  float64 
     36  weekday_is_saturday            39644 non-null  float64 
     37  weekday_is_sunday              39644 non-null  float64 
     38  is_weekend                     39644 non-null  float64 
     39  LDA_00                         39644 non-null  float64 
     40  LDA_01                         39644 non-null  float64 
     41  LDA_02                         39644 non-null  float64 
     42  LDA_03                         39644 non-null  float64 
     43  LDA_04                         39644 non-null  float64 
     44  global_subjectivity            39644 non-null  float64 
     45  global_sentiment_polarity      39644 non-null  float64 
     46  global_rate_positive_words     39644 non-null  float64 
     47  global_rate_negative_words     39644 non-null  float64 
     48  rate_positive_words            39644 non-null  float64 
     49  rate_negative_words            39644 non-null  float64 
     50  avg_positive_polarity          39644 non-null  float64 
     51  min_positive_polarity          39644 non-null  float64 
     52  max_positive_polarity          39644 non-null  float64 
     53  avg_negative_polarity          39644 non-null  float64 
     54  min_negative_polarity          39644 non-null  float64 
     55  max_negative_polarity          39644 non-null  float64 
     56  title_subjectivity             39644 non-null  float64 
     57  title_sentiment_polarity       39644 non-null  float64 
     58  abs_title_subjectivity         39644 non-null  float64 
     59  abs_title_sentiment_polarity   39644 non-null  float64 
     60  shares                         39644 non-null  int64   
     61  log_shares                     39644 non-null  float64 
     62  boxcox_shares                  39644 non-null  float64 
     63  data_channel                   39644 non-null  object  
     64  zscore                         39644 non-null  float64 
     65  high_low_category              39644 non-null  object  
     66  shares_category                39644 non-null  category
    dtypes: category(1), float64(62), int64(1), object(3)
    memory usage: 20.0+ MB
    


```python
# List of feature names to convert to boolean
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

# Convert each feature to boolean type
for col in bool_features:
    if col in df.columns:
        df[col] = df[col].astype(bool)
    else:
        print(f"Warning: Column '{col}' not found in DataFrame.")

```


```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 39644 entries, 0 to 39643
    Data columns (total 67 columns):
     #   Column                         Non-Null Count  Dtype   
    ---  ------                         --------------  -----   
     0   url                            39644 non-null  object  
     1   timedelta                      39644 non-null  float64 
     2   n_tokens_title                 39644 non-null  float64 
     3   n_tokens_content               39644 non-null  float64 
     4   n_unique_tokens                39644 non-null  float64 
     5   n_non_stop_words               39644 non-null  float64 
     6   n_non_stop_unique_tokens       39644 non-null  float64 
     7   num_hrefs                      39644 non-null  float64 
     8   num_self_hrefs                 39644 non-null  float64 
     9   num_imgs                       39644 non-null  float64 
     10  num_videos                     39644 non-null  float64 
     11  average_token_length           39644 non-null  float64 
     12  num_keywords                   39644 non-null  float64 
     13  data_channel_is_lifestyle      39644 non-null  bool    
     14  data_channel_is_entertainment  39644 non-null  bool    
     15  data_channel_is_bus            39644 non-null  bool    
     16  data_channel_is_socmed         39644 non-null  bool    
     17  data_channel_is_tech           39644 non-null  bool    
     18  data_channel_is_world          39644 non-null  bool    
     19  kw_min_min                     39644 non-null  float64 
     20  kw_max_min                     39644 non-null  float64 
     21  kw_avg_min                     39644 non-null  float64 
     22  kw_min_max                     39644 non-null  float64 
     23  kw_max_max                     39644 non-null  float64 
     24  kw_avg_max                     39644 non-null  float64 
     25  kw_min_avg                     39644 non-null  float64 
     26  kw_max_avg                     39644 non-null  float64 
     27  kw_avg_avg                     39644 non-null  float64 
     28  self_reference_min_shares      39644 non-null  float64 
     29  self_reference_max_shares      39644 non-null  float64 
     30  self_reference_avg_sharess     39644 non-null  float64 
     31  weekday_is_monday              39644 non-null  bool    
     32  weekday_is_tuesday             39644 non-null  bool    
     33  weekday_is_wednesday           39644 non-null  bool    
     34  weekday_is_thursday            39644 non-null  bool    
     35  weekday_is_friday              39644 non-null  bool    
     36  weekday_is_saturday            39644 non-null  bool    
     37  weekday_is_sunday              39644 non-null  bool    
     38  is_weekend                     39644 non-null  bool    
     39  LDA_00                         39644 non-null  float64 
     40  LDA_01                         39644 non-null  float64 
     41  LDA_02                         39644 non-null  float64 
     42  LDA_03                         39644 non-null  float64 
     43  LDA_04                         39644 non-null  float64 
     44  global_subjectivity            39644 non-null  float64 
     45  global_sentiment_polarity      39644 non-null  float64 
     46  global_rate_positive_words     39644 non-null  float64 
     47  global_rate_negative_words     39644 non-null  float64 
     48  rate_positive_words            39644 non-null  float64 
     49  rate_negative_words            39644 non-null  float64 
     50  avg_positive_polarity          39644 non-null  float64 
     51  min_positive_polarity          39644 non-null  float64 
     52  max_positive_polarity          39644 non-null  float64 
     53  avg_negative_polarity          39644 non-null  float64 
     54  min_negative_polarity          39644 non-null  float64 
     55  max_negative_polarity          39644 non-null  float64 
     56  title_subjectivity             39644 non-null  float64 
     57  title_sentiment_polarity       39644 non-null  float64 
     58  abs_title_subjectivity         39644 non-null  float64 
     59  abs_title_sentiment_polarity   39644 non-null  float64 
     60  shares                         39644 non-null  int64   
     61  log_shares                     39644 non-null  float64 
     62  boxcox_shares                  39644 non-null  float64 
     63  data_channel                   39644 non-null  object  
     64  zscore                         39644 non-null  float64 
     65  high_low_category              39644 non-null  object  
     66  shares_category                39644 non-null  category
    dtypes: bool(14), category(1), float64(48), int64(1), object(3)
    memory usage: 16.3+ MB
    


```python
train = df.drop(['url', 'timedelta','shares_category','shares'], axis=1)
```


```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Assume df is already defined and loaded with data

# 1. Calculate and print the missing values for each feature
missing_values = train.isnull().sum()
print("Missing values in each feature:")
print(missing_values[missing_values > 0])  # Only print features with missing values

# 3. Heatmap for Missing Values
plt.figure(figsize=(12, 8))
sns.heatmap(train.isnull(), cbar=False, yticklabels=False, cmap='viridis')
plt.title("Heatmap of Missing Values")
plt.show()

```

    Missing values in each feature:
    Series([], dtype: int64)
    


    
![png](EDA%20and%20Feature%20Engineering_files/EDA%20and%20Feature%20Engineering_25_1.png)
    



```python
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import math


df_sample = train

# Set Seaborn style (optional)
sns.set(style="whitegrid")

# Define how many plots per page
plots_per_page = 9
n_features = len(df_sample.columns)
n_pages = math.ceil(n_features / plots_per_page)
columns = df_sample.columns  # List of column names

# Loop over pages
for page in range(n_pages):
    # Create a 3x3 grid of subplots
    fig, axes = plt.subplots(3, 3, figsize=(20, 18))
    axes = axes.flatten()  # Flatten the grid for easy iteration
    
    # Loop over each subplot on the current page
    for i in range(plots_per_page):
        index = page * plots_per_page + i
        if index >= n_features:
            # Hide any extra subplots on the last page
            axes[i].axis('off')
        else:
            col = columns[index]
            ax = axes[i]
            
            # Plot numeric features with histogram; categorical with countplot
            if pd.api.types.is_numeric_dtype(df_sample[col]):
                sns.histplot(df_sample[col], kde=False, bins=20, color='skyblue', ax=ax)
                ax.set_xlabel(col)
                ax.set_ylabel('Frequency')
                ax.set_title(f'Feature "{col}" Distribution (Numeric)')
            else:
                order = df_sample[col].value_counts().index
                sns.countplot(x=df_sample[col], order=order, palette='viridis', ax=ax)
                ax.set_xlabel(col)
                ax.set_ylabel('Count')
                ax.set_title(f'Feature "{col}" Distribution (Categorical)')
                ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.show()

```


    
![png](EDA%20and%20Feature%20Engineering_files/EDA%20and%20Feature%20Engineering_26_0.png)
    



    
![png](EDA%20and%20Feature%20Engineering_files/EDA%20and%20Feature%20Engineering_26_1.png)
    



    
![png](EDA%20and%20Feature%20Engineering_files/EDA%20and%20Feature%20Engineering_26_2.png)
    



    
![png](EDA%20and%20Feature%20Engineering_files/EDA%20and%20Feature%20Engineering_26_3.png)
    



    
![png](EDA%20and%20Feature%20Engineering_files/EDA%20and%20Feature%20Engineering_26_4.png)
    



    
![png](EDA%20and%20Feature%20Engineering_files/EDA%20and%20Feature%20Engineering_26_5.png)
    



    
![png](EDA%20and%20Feature%20Engineering_files/EDA%20and%20Feature%20Engineering_26_6.png)
    



```python
import pandas as pd
import numpy as np

skew_values = {}
for col in train.columns:
    if pd.api.types.is_numeric_dtype(train[col]):
        skew_values[col] = train[col].skew()
    else:
        skew_values[col] = 'N/A' 

for feature, skew_val in skew_values.items():
    print(f"{feature}: {skew_val}")

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
    


```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import PowerTransformer

# Boolean features (skip processing)
boolean_features = [
    'data_channel_is_lifestyle',
    'data_channel_is_entertainment',
    'weekday_is_monday',
    'weekday_is_tuesday',
    'weekday_is_wednesday',
    'weekday_is_thursday',
    'weekday_is_friday',
    'weekday_is_saturday',
    'weekday_is_sunday',
    'is_weekend',
    'data_channel_is_bus',
    'data_channel_is_socmed',
    'data_channel_is_tech',
    'data_channel_is_world'
]

# Predefined skewness values for each feature
skew_values = {
    'n_tokens_title': 0.16532037674928027,
    'n_tokens_content': 2.9454219387867084,
    'n_unique_tokens': 198.65511559825592,
    'n_non_stop_words': 198.7924453768874,
    'n_non_stop_unique_tokens': 198.44329440926512,
    'num_hrefs': 4.013494828201318,
    'num_self_hrefs': 5.172751105757634,
    'num_imgs': 3.9465958446535474,
    'num_videos': 7.0195327862958665,
    'average_token_length': -4.57601155020474,
    'num_keywords': -0.14725125199950523,
    'kw_min_min': 2.3749472801825444,
    'kw_max_min': 35.32843373115432,
    'kw_avg_min': 31.306108102660584,
    'kw_min_max': 10.386371634782769,
    'kw_max_max': -2.6449817621966782,
    'kw_avg_max': 0.6243096463608944,
    'kw_min_avg': 0.4679758464905322,
    'kw_max_avg': 16.41166955537124,
    'kw_avg_avg': 5.760177291618559,
    'self_reference_min_shares': 26.264364160300094,
    'self_reference_max_shares': 13.870849049433598,
    'self_reference_avg_sharess': 17.9140933776756,
    'LDA_00': 1.5674632332004765,
    'LDA_01': 2.0867218234169407,
    'LDA_02': 1.311694902028395,
    'LDA_03': 1.2387159863782728,
    'LDA_04': 1.1731294759766238,
    'global_subjectivity': -1.3726888305603973,
    'global_sentiment_polarity': 0.10545709665820545,
    'global_rate_positive_words': 0.32304661115048916,
    'global_rate_negative_words': 1.491917309190822,
    'rate_positive_words': -1.423105853002299,
    'rate_negative_words': 0.4072406539941212,
    'avg_positive_polarity': -0.7247949503201233,
    'min_positive_polarity': 3.0404677374643283,
    'max_positive_polarity': -0.9397564591253907,
    'avg_negative_polarity': -0.55164402900095,
    'min_negative_polarity': -0.07315481617331099,
    'max_negative_polarity': -3.4597470578480207,
    'title_subjectivity': 0.816084749635643,
    'title_sentiment_polarity': 0.39610883665169594,
    'abs_title_subjectivity': -0.6241493828840421,
    'abs_title_sentiment_polarity': 1.7041934399140888,
    'log_shares': 1.0264770065011177,
    'boxcox_shares': -0.2239496211569689,
    'zscore': 33.96388487571418
}

# Handling features based on skewness
for feature, skewness in skew_values.items():
    if feature in boolean_features:
        continue  # Skip boolean features

    if skewness > 10:  # Extremely high skewness
        print(f"Applying Yeo-Johnson transformation to {feature} (skewness={skewness:.2f})")
        transformer = PowerTransformer(method='yeo-johnson', standardize=False)
        df[feature] = transformer.fit_transform(df[[feature]])
    elif 2 < skewness <= 10:  # Moderate skewness
        print(f"Applying square root transformation to {feature} (skewness={skewness:.2f})")
        df[feature] = np.sqrt(df[feature])
    elif skewness < -10:  # Extreme negative skewness
        print(f"Applying inverse transformation to {feature} (skewness={skewness:.2f})")
        df[feature] = -1 / (df[feature] + 1e-5)  # Avoid division by zero
    elif -10 < skewness < -2:  # Moderate negative skewness
        print(f"Applying square root transformation to {feature} (skewness={skewness:.2f})")
        df[feature] = -np.sqrt(-df[feature])  # Use negative square root for left-skewed data
    else:
        print(f"No transformation applied to {feature} (skewness={skewness:.2f})")

# Final check
print("\nFeature processing completed!")

```

    No transformation applied to n_tokens_title (skewness=0.17)
    Applying square root transformation to n_tokens_content (skewness=2.95)
    Applying Yeo-Johnson transformation to n_unique_tokens (skewness=198.66)
    Applying Yeo-Johnson transformation to n_non_stop_words (skewness=198.79)
    Applying Yeo-Johnson transformation to n_non_stop_unique_tokens (skewness=198.44)
    Applying square root transformation to num_hrefs (skewness=4.01)
    Applying square root transformation to num_self_hrefs (skewness=5.17)
    Applying square root transformation to num_imgs (skewness=3.95)
    Applying square root transformation to num_videos (skewness=7.02)
    Applying square root transformation to average_token_length (skewness=-4.58)
    No transformation applied to num_keywords (skewness=-0.15)
    Applying square root transformation to kw_min_min (skewness=2.37)
    Applying Yeo-Johnson transformation to kw_max_min (skewness=35.33)
    Applying Yeo-Johnson transformation to kw_avg_min (skewness=31.31)
    Applying Yeo-Johnson transformation to kw_min_max (skewness=10.39)
    Applying square root transformation to kw_max_max (skewness=-2.64)
    No transformation applied to kw_avg_max (skewness=0.62)
    No transformation applied to kw_min_avg (skewness=0.47)
    Applying Yeo-Johnson transformation to kw_max_avg (skewness=16.41)
    Applying square root transformation to kw_avg_avg (skewness=5.76)
    Applying Yeo-Johnson transformation to self_reference_min_shares (skewness=26.26)
    Applying Yeo-Johnson transformation to self_reference_max_shares (skewness=13.87)
    Applying Yeo-Johnson transformation to self_reference_avg_sharess (skewness=17.91)
    No transformation applied to LDA_00 (skewness=1.57)
    Applying square root transformation to LDA_01 (skewness=2.09)
    No transformation applied to LDA_02 (skewness=1.31)
    No transformation applied to LDA_03 (skewness=1.24)
    No transformation applied to LDA_04 (skewness=1.17)
    No transformation applied to global_subjectivity (skewness=-1.37)
    No transformation applied to global_sentiment_polarity (skewness=0.11)
    No transformation applied to global_rate_positive_words (skewness=0.32)
    No transformation applied to global_rate_negative_words (skewness=1.49)
    No transformation applied to rate_positive_words (skewness=-1.42)
    No transformation applied to rate_negative_words (skewness=0.41)
    No transformation applied to avg_positive_polarity (skewness=-0.72)
    Applying square root transformation to min_positive_polarity (skewness=3.04)
    No transformation applied to max_positive_polarity (skewness=-0.94)
    No transformation applied to avg_negative_polarity (skewness=-0.55)
    No transformation applied to min_negative_polarity (skewness=-0.07)
    Applying square root transformation to max_negative_polarity (skewness=-3.46)
    No transformation applied to title_subjectivity (skewness=0.82)
    No transformation applied to title_sentiment_polarity (skewness=0.40)
    No transformation applied to abs_title_subjectivity (skewness=-0.62)
    No transformation applied to abs_title_sentiment_polarity (skewness=1.70)
    No transformation applied to log_shares (skewness=1.03)
    No transformation applied to boxcox_shares (skewness=-0.22)
    Applying Yeo-Johnson transformation to zscore (skewness=33.96)
    
    Feature processing completed!
    


```python
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import math


# Set Seaborn style (optional)
sns.set(style="whitegrid")

# Define how many plots per page
plots_per_page = 9
n_features = len(df.columns)
n_pages = math.ceil(n_features / plots_per_page)
columns = df.columns  # List of column names

# Loop over pages
for page in range(n_pages):
    # Create a 3x3 grid of subplots
    fig, axes = plt.subplots(3, 3, figsize=(20, 18))
    axes = axes.flatten()  # Flatten the grid for easy iteration
    
    # Loop over each subplot on the current page
    for i in range(plots_per_page):
        index = page * plots_per_page + i
        if index >= n_features:
            # Hide any extra subplots on the last page
            axes[i].axis('off')
        else:
            col = columns[index]
            ax = axes[i]
            
            # Plot numeric features with histogram; categorical with countplot
            if pd.api.types.is_numeric_dtype(df[col]):
                sns.histplot(df[col], kde=False, bins=20, color='skyblue', ax=ax)
                ax.set_xlabel(col)
                ax.set_ylabel('Frequency')
                ax.set_title(f'Feature "{col}" Distribution (Numeric)')
            else:
                order = df[col].value_counts().index
                sns.countplot(x=df[col], order=order, palette='viridis', ax=ax)
                ax.set_xlabel(col)
                ax.set_ylabel('Count')
                ax.set_title(f'Feature "{col}" Distribution (Categorical)')
                ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.show()

```


    ---------------------------------------------------------------------------

    KeyboardInterrupt                         Traceback (most recent call last)

    Cell In[30], line 43
         41 else:
         42     order = df[col].value_counts().index
    ---> 43     sns.countplot(x=df[col], order=order, palette='viridis', ax=ax)
         44     ax.set_xlabel(col)
         45     ax.set_ylabel('Count')
    

    File d:\Anaconda\Lib\site-packages\seaborn\categorical.py:2675, in countplot(data, x, y, hue, order, hue_order, orient, color, palette, saturation, fill, hue_norm, stat, width, dodge, gap, log_scale, native_scale, formatter, legend, ax, **kwargs)
       2671     p.plot_data[count_axis] /= len(p.plot_data) / denom
       2673 aggregator = EstimateAggregator("sum", errorbar=None)
    -> 2675 p.plot_bars(
       2676     aggregator=aggregator,
       2677     dodge=dodge,
       2678     width=width,
       2679     gap=gap,
       2680     color=color,
       2681     fill=fill,
       2682     capsize=0,
       2683     err_kws={},
       2684     plot_kws=kwargs,
       2685 )
       2687 p._add_axis_labels(ax)
       2688 p._adjust_cat_axis(ax, axis=p.orient)
    

    File d:\Anaconda\Lib\site-packages\seaborn\categorical.py:1293, in _CategoricalPlotter.plot_bars(self, aggregator, dodge, gap, width, fill, color, capsize, err_kws, plot_kws)
       1290     agg_data["width"] *= 1 - gap
       1292 agg_data["edge"] = agg_data[self.orient] - agg_data["width"] / 2
    -> 1293 self._invert_scale(ax, agg_data)
       1295 if self.orient == "x":
       1296     bar_func = ax.bar
    

    File d:\Anaconda\Lib\site-packages\seaborn\categorical.py:419, in _CategoricalPlotter._invert_scale(self, ax, data, vars)
        417 for suf in ["", "min", "max"]:
        418     if (col := f"{var}{suf}") in data:
    --> 419         data[col] = inv(data[col])
    

    File d:\Anaconda\Lib\site-packages\pandas\core\frame.py:4311, in DataFrame.__setitem__(self, key, value)
       4308     self._setitem_array([key], value)
       4309 else:
       4310     # set column
    -> 4311     self._set_item(key, value)
    

    File d:\Anaconda\Lib\site-packages\pandas\core\frame.py:4538, in DataFrame._set_item(self, key, value)
       4535             value = np.tile(value, (len(existing_piece.columns), 1)).T
       4536             refs = None
    -> 4538 self._set_item_mgr(key, value, refs)
    

    File d:\Anaconda\Lib\site-packages\pandas\core\frame.py:4490, in DataFrame._set_item_mgr(self, key, value, refs)
       4488     self._mgr.insert(len(self._info_axis), key, value, refs)
       4489 else:
    -> 4490     self._iset_item_mgr(loc, value, refs=refs)
       4492 # check if we are modifying a copy
       4493 # try to set first as we want an invalid
       4494 # value exception to occur first
       4495 if len(self):
    

    File d:\Anaconda\Lib\site-packages\pandas\core\frame.py:4478, in DataFrame._iset_item_mgr(self, loc, value, inplace, refs)
       4470 def _iset_item_mgr(
       4471     self,
       4472     loc: int | slice | np.ndarray,
       (...)
       4476 ) -> None:
       4477     # when called from _set_item_mgr loc can be anything returned from get_loc
    -> 4478     self._mgr.iset(loc, value, inplace=inplace, refs=refs)
       4479     self._clear_item_cache()
    

    File d:\Anaconda\Lib\site-packages\pandas\core\internals\managers.py:1211, in BlockManager.iset(self, loc, value, inplace, refs)
       1208     self._blknos[unfit_idxr] = len(self.blocks)
       1209     self._blklocs[unfit_idxr] = np.arange(unfit_count)
    -> 1211 self.blocks += tuple(new_blocks)
       1213 # Newly created block's dtype may already be present.
       1214 self._known_consolidated = False
    

    KeyboardInterrupt: 



```python
skew_values = {}
for col in train.columns:
    if pd.api.types.is_numeric_dtype(train[col]):
        skew_values[col] = df[col].skew()
    else:
        skew_values[col] = 'N/A' 

for feature, skew_val in skew_values.items():
    print(f"{feature}: {skew_val}")

```

    n_tokens_title: 0.16532037674928027
    n_tokens_content: 0.6532703475027776
    n_unique_tokens: -0.4952612309048895
    n_non_stop_words: 0.5507252369342349
    n_non_stop_unique_tokens: -0.2426394837700783
    num_hrefs: 1.0288399557580805
    num_self_hrefs: 0.6325282374409829
    num_imgs: 1.510209402249247
    num_videos: 2.69996521216398
    average_token_length: 0.0
    num_keywords: -0.14725125199950523
    data_channel_is_lifestyle: 3.9930191433554167
    data_channel_is_entertainment: 1.68358480940472
    data_channel_is_bus: 1.8768701859879158
    data_channel_is_socmed: 3.758879630973088
    data_channel_is_tech: 1.6199757646890423
    data_channel_is_world: 1.4051693841208097
    kw_min_min: 0.9801816664720001
    kw_max_min: 0.4685096472876654
    kw_avg_min: 0.5720539861620555
    kw_min_max: -0.06416892401216162
    kw_max_max: 0.0
    kw_avg_max: 0.6243096463608944
    kw_min_avg: 0.4679758464905322
    kw_max_avg: 0.7665264788447039
    kw_avg_avg: 1.3878735195269234
    self_reference_min_shares: -0.029695874578419528
    self_reference_max_shares: -0.10076895481923329
    self_reference_avg_sharess: -0.09010658548926888
    weekday_is_monday: 1.7759082442285052
    weekday_is_tuesday: 1.6105470619092879
    weekday_is_wednesday: 1.6009709768881089
    weekday_is_thursday: 1.6370700482983118
    weekday_is_friday: 2.030304835180609
    weekday_is_saturday: 3.6370857599701125
    weekday_is_sunday: 3.3999273763003046
    is_weekend: 2.18850033431371
    LDA_00: 1.5674632332004765
    LDA_01: 1.5496121678485797
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
    min_positive_polarity: 0.44901772425609876
    max_positive_polarity: -0.9397564591253907
    avg_negative_polarity: -0.55164402900095
    min_negative_polarity: -0.07315481617331099
    max_negative_polarity: -0.45426231980662274
    title_subjectivity: 0.816084749635643
    title_sentiment_polarity: 0.39610883665169594
    abs_title_subjectivity: -0.6241493828840421
    abs_title_sentiment_polarity: 1.7041934399140888
    log_shares: 1.0264770065011177
    boxcox_shares: -0.22394970896715033
    data_channel: N/A
    zscore: 0.6959825995828395
    high_low_category: N/A
    


```python

```
