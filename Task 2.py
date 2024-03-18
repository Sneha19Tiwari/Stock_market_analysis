#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd


# In[6]:


apple = pd.read_csv(r'C:\Users\91823\Downloads\archive (2)\stocks\AMZN.csv')
facebook = pd.read_csv(r'C:\Users\91823\Downloads\archive (2)\stocks\FB.csv')
google = pd.read_csv(r"C:\Users\91823\Downloads\archive (2)\stocks\GOOGL.csv")
nvidia = pd.read_csv(r"C:\Users\91823\Downloads\archive (2)\stocks\NVDA.csv")
tesla = pd.read_csv(r"C:\Users\91823\Downloads\archive (2)\stocks\TSLA.csv")
twitter = pd.read_csv(r"C:\Users\91823\Downloads\archive (2)\stocks\TWTR.csv")


# In[7]:


apple.head()


# In[8]:


facebook.head()


# In[9]:


google.head()


# In[10]:


nvidia.head()


# In[11]:


tesla.head()


# In[12]:


twitter.head()


# In[13]:


dfs = [apple, facebook, google, nvidia, tesla, twitter]


# In[14]:


for df in dfs:
    df['MA50'] = df.Close.rolling(50).mean()
    df['MA200'] = df.Close.rolling(200).mean()


# In[15]:


apple.head(200)


# In[16]:


for df in dfs:
    df['Previous day close price'] = df.Close.shift(1)


# In[17]:


apple.head()


# In[18]:


for df in dfs:
    df['Change in price'] = df['Close'] - df['Previous day close price']


# In[19]:


apple.head()


# In[20]:


for df in dfs:
    df['Percent change in price'] = df.Close.pct_change()


# In[21]:


apple.head()


# In[22]:


for df in dfs:
    df['Previous day volume'] = df.Volume.shift(1)


# In[23]:


apple.head()


# In[24]:


for df in dfs:
    df['Change in volume'] = df['Volume'] - df['Previous day volume']


# In[25]:


apple.head()


# In[26]:


for df in dfs:
    df['Percent change in volume'] = df.Volume.pct_change()


# In[27]:


apple.head()


# In[28]:


apple.to_csv('Apple.csv')
facebook.to_csv('Facebook.csv')
google.to_csv('Google.csv')
nvidia.to_csv('Nvidia.csv')
tesla.to_csv('Tesla.csv')
twitter.to_csv('Twitter.csv')


# In[ ]:




