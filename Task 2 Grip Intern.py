#!/usr/bin/env python
# coding: utf-8

# # Task 2: Linear Regression with Python Scikit Learn

# ## Simple Linear Regression
# In this regression task we will predict the percentage of marks that a student is expected to score based upon the number of hours they studied. This is a simple linear regression task as it involves just two variables.

# In[2]:


import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt  
get_ipython().run_line_magic('matplotlib', 'inline')


# Importing the data

# In[3]:


url = "http://bit.ly/w-data"
data = pd.read_csv(url)
print("Data imported")

data.head(10)


# Plotting data on the 2D graph

# In[4]:


data.plot(x='Hours', y='Scores', style='x')  
plt.title('Hours vs Percentage')  
plt.xlabel('Hours Studied')  
plt.ylabel('Percentage Score')  
plt.show()


# Preparing Data

# In[5]:


X = data.iloc[:, :-1].values  
y = data.iloc[:, 1].values  


# Splitting data into training and test case

# In[6]:


from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                            test_size=0.2, random_state=0) 


# Training Algorithm

# In[7]:


from sklearn.linear_model import LinearRegression  
regressor = LinearRegression()  
regressor.fit(X_train, y_train) 

print("Training complete.")


# In[8]:


line = regressor.coef_*X+regressor.intercept_

# Plotting the test data
plt.scatter(X, y)
plt.plot(X, line);
plt.show()


# Prediction time

# In[9]:


print(X_test) # Testing data 
y_pred = regressor.predict(X_test) # Predicting the scores


# Comparing Actual Aginst Predicted

# In[10]:


df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
df


# In[11]:


hours = [[9.25]]
own_pred = regressor.predict(hours)
print("No of Hours = {}".format(hours))
print("Predicted Score = {}".format(own_pred[0]))


# Evaluating Model

# In[13]:


from sklearn import metrics  
print('Mean Absolute Error:', 
      metrics.mean_absolute_error(y_test, y_pred)) 


# Completed.
