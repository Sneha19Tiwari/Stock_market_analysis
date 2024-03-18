#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from datetime import datetime, timedelta


# In[2]:


# Load historical stock price data
data = pd.read_csv(r'C:\Users\91823\Downloads\datacamp_workspace_export_2024-03-14 11_07_56.csv')


# In[3]:


# Assuming the dataset has 'Date' and 'Close' colum

# Feature engineering
# For simplicity, let's use only the 'Close' price as the feature
X = data['Close'].values.reshape(-1, 1)

# Target variable: next day's closing price
y = data['Close'].shift(-1).fillna(method='ffill').values  # Using the next day's closing price as the target


# In[4]:


# Splitting data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[5]:


# Training the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)


# In[6]:


# Making predictions for the next 3 days
last_day_close = X[-1].reshape(1, -1)  # Using the last available closing price for prediction
next_three_days_predictions = []
for i in range(3):
    next_day_prediction = model.predict(last_day_close)[0]
    next_three_days_predictions.append(next_day_prediction)
    last_day_close = np.array([[next_day_prediction]])  # Update for the next day's prediction


# In[7]:


# Printing the predicted prices for the next 3 days
print("Predicted Prices for the Next 3 Days:")
for i, price in enumerate(next_three_days_predictions):
    print(f"Day {i+1}: {price:.2f}")


# In[8]:


# Get the last date in the dataset
data['Date'] = pd.to_datetime(data['Date'])
last_date = data['Date'].iloc[-1]

# Generate dates for the next 3 days
next_3_days = [last_date + timedelta(days=i) for i in range(1, 4)]


# In[9]:


# Visualizing the predicted prices
plt.plot(data['Date'], data['Close'], label='Actual Prices')
plt.title('Predicted Stock Prices for the Next 3 Days')
plt.xlabel('Date')
plt.ylabel('Closing Price')
plt.xticks(rotation=45)
plt.plot([data['Date'].iloc[-1]] + next_3_days, [data['Close'].iloc[-1]] + next_three_days_predictions, marker='o', linestyle='-', label='Predicted Prices')
plt.legend()
plt.tight_layout()
plt.show()


# In[10]:


get_ipython().system('pip install vaderSentiment')
get_ipython().system('pip install newsapi-python')


# In[11]:


get_ipython().system('pip install tweepy')


# In[ ]:





# In[1]:


import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Initialize VADER sentiment analyzer
analyzer = SentimentIntensityAnalyzer()
news_data = pd.read_csv(r"C:\Users\91823\Downloads\archive (1)\raw_partner_headlines.csv", encoding='ISO-8859-1')

analyzer = SentimentIntensityAnalyzer()

# Perform sentiment analysis on news headlines using VADER
sentiment_scores = []

for headline in news_data['headline']:  # Replace 'headline' with the actual column name
    sentiment_score = analyzer.polarity_scores(headline)['compound']
    sentiment_scores.append(sentiment_score)
# Add sentiment scores to the news dataset    
news_data['sentiment_score'] = sentiment_scores

# Display the news dataset with sentiment scores
print(news_data)


# In[ ]:




