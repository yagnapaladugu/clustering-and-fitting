#!/usr/bin/env python
# coding: utf-8

# In[3]:

import numpy as num 
import pandas as pan
import matplotlib.pyplot as plot
import seaborn as sns ## importing libraries


# In[38]:


import warnings
warnings.filterwarnings('ignore')


# In[27]:


df=pan.read_csv('8fbda349-8b66-477c-bac2-d9723769aa19_Data.csv')


# In[40]:


df.isna().sum()


# In[28]:


df = df.dropna()
print(df)


# In[29]:


df.shape


# In[30]:


df.describe()


# In[44]:


df.info()


# In[45]:


from sklearn.preprocessing import LabelEncoder


# In[46]:


X = df

y = df['Series Name'] ##  separating variables


# In[47]:


le = LabelEncoder()

X['Series Name'] = le.fit_transform(X['Series Name']) ## converting object type into numerical type


# In[48]:


X


# In[14]:


y


# ## K-means Clustering

# In[49]:


from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import matplotlib.pyplot as plot

# Generating sample data
X, y = make_blobs(n_samples=300, n_features=2, centers=3, cluster_std=0.5, shuffle=True, random_state=0)

# Implementing k-means clustering
kmeans_clustering = KMeans(n_clusters=3, init='k-means++', max_iter=300, n_init=10, random_state=0)
pred_y = kmeans_clustering.fit_predict(X)

# Plotting the data points colored by cluster
plot.scatter(X[:,0], X[:,1], c=pred_y, cmap='viridis')

# Plotting the cluster centers
plot.scatter(kmeans_clustering.cluster_centers_[:, 0], kmeans_clustering.cluster_centers_[:, 1], s=300, c='red')

# Adding labels and title
plot.xlabel('X')
plot.ylabel('Y')
plot.title('Curve between predictions and number of clusters')

# Showing plot
plot.show()


# ## Pie chart of fraction

# In[50]:


# Sample data
data = [1247419, 1229374, 1200129, 1170884, 1141639, 1112394, 1083817.597, 1083817.597, 1083817.597]
keys=['2012 [YR2012]','2013 [YR2013]','2014 [YR2014]','2015 [YR2015]','2016 [YR2016]','2017 [YR2017]','2018 [YR2018]','2019 [YR2019]','2020 [YR2020]']
# define Seaborn color palette to use
palette_color = sns.color_palette('dark')
  
# plotting data on chart
plot.pie(data, labels=keys, colors=palette_color,
        autopct='%.0f%%')
  
# displaying chart
plot.show()


# ## Bar chart of 10 years interval

# In[51]:


# data for the bar plot
years = [2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021]
values = [1.6571744743664, 1.89991945754679, 1.63213203434928, 2.88973541352409, 2.51367533486328, 1.92059051131218, 1.15610643376958, 1.57064146480711, 0.774660884815787, 2.00903976421976]

# creating the bar plot
plot.bar(years, values)

# adding axis labels and a title
plot.xlabel('Years')
plot.ylabel('Values')
plot.title('Values by 10-year Intervals')

# displaying the plot
plot.show()


# ## Log scale 

# In[55]:


# Creating some data
x = ['Afganisthan','Colombia', 'Argentina', 'Austraila', "Belgium", 'Cambodia', 'Chile','Croatia', 'Denmark', 'Greece']
y = [383560, 482428.225, 1083817.597, 3557750, 13646.093, 57890, 157100, 15050, 26199.87, 58671.88]

# Plotting the data
plot.plot(x, y)

plot.xticks(rotation=45, horizontalalignment="center")

# Setting the y-axis to a log scale
plot.yscale("log")

plot.title("Log scale to compare CO2 emissions between 10 countries in 2020")

# displaying the plot
plot.show()


# ## Normalise to 1 or 100 for 1 year

# In[15]:


import pandas as pd
df = pd.DataFrame({'value':[379100, 1200129, 2347174, 13313, 579130]})
df["normalized"] = df["value"].div(365).mul(100)


# In[16]:


df.normalized


# ## Divide by the average

# In[20]:


import pandas as pd
df = pd.DataFrame({'value':[379100, 1200129, 2347174, 13313, 579130]})
df["normalized"] = df["value"].div(df["value"].mean())


# In[18]:


df.normalized


# In[31]:


df = df.replace('..', num.nan)
df.isnull().sum().any()
df = df.dropna()


# In[34]:


from sklearn import preprocessing


# In[36]:


label_encoder = preprocessing.LabelEncoder()



# In[37]:





# In[39]:




# In[40]:


df.head()


# ## Curve fit and best fitting plot

# In[46]:





# In[47]:


X


# In[48]:


y


# In[49]:


from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import numpy as np

# Defining exponential growth model function
def exp_growth(x, a, b):
    return a * np.exp(b * x)

# Generating data points
X = np.linspace(0, 10, 50)
y = exp_growth(X, 2, 0.5) + np.random.normal(0, 0.2, 50)

def err_ranges(params, covariance, sigma): # using err_ranges function
    # Compute the standard deviation for each parameter
    stdevs = np.sqrt(np.diag(covariance))
    
    # Compute the lower and upper limits for each parameter
    lower = params - sigma * stdevs
    upper = params + sigma * stdevs
    
    return lower, upper

# Fitting data to exponential growth model
params, covariance = curve_fit(exp_growth, X, y)

# Estimating lower and upper limits of the confidence range
lower, upper = err_ranges(params, covariance, 2)

# Plotting data and best-fitting function
plt.scatter(X, y)
plt.plot(X, exp_growth(X, params[0], params[1]), 'r-')

# Plotting lower and upper limits of the confidence range
plt.fill_between(X, exp_growth(X, lower[0], lower[1]), exp_growth(X, upper[0], upper[1]), color='gray', alpha=0.5)

plt.show()

