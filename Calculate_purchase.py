#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd

# Load data from excel file
data = pd.read_excel('GHL_Form5.xlsx', sheet_name = 'Sheet1')


# In[2]:


data


# In[3]:


# Extract relevant columns
age = data['Age']
budget = data['Budget']
purchase_history = data['purchase_history']
reviews = data['reviews']
Gender = data['Gender']
price = data['Price']


# In[4]:


# Coefficients for each input
coefficients = {'age': 0.1, 'budget': 0.2, 'purchase_history': 0.3, 'reviews': 0.4, 'gender': 0.1, 'price': 0.05}


# In[5]:


# Add a new column to the dataframe to store the output percentage likelihood
data['percentage_likelihood'] = 0


# In[6]:


# Calculate the percentage likelihood using the coefficients
for index, row in data.iterrows():
    percentage_likelihood = (row['Age'] * coefficients['age'] + row['Budget'] * coefficients['budget'] + row['purchase_history'] * coefficients['purchase_history'] + row['reviews'] * coefficients['reviews'] + (1 if row['Gender'] == 'Male' else 0) * coefficients['gender'] + row['Price'] * coefficients['price']) * 100
    data.at[index, 'percentage_likelihood'] = percentage_likelihood


# In[7]:


# Print the output data
print(data)


# In[8]:


data


# In[9]:


import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib


# In[10]:


# Alternatively, you can fill the column with specific values based on the existing data
data["income_level"] = data.apply(lambda x: "low" if x["percentage_likelihood"] < 50000 else "high", axis=1)


# In[11]:


data


# In[12]:


data['Gender'] = data['Gender'].map({'Male':1,'Female':0})


# In[13]:


data


# In[14]:


# Define the features
X = data[['Age', 'Gender', 'purchase_history', 'reviews', 'Price']]


# In[15]:


# Define the target variable
y = data['income_level']


# In[16]:


# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[17]:


# Create a logistic regression model
log_reg = LogisticRegression()


# In[19]:


log_reg.fit(X_train, y_train)


# In[20]:


# Make predictions on the test data
y_pred = log_reg.predict(X_test)


# In[21]:


# Calculate the accuracy of the model
acc = accuracy_score(y_test, y_pred)


# In[22]:


# Print the accuracy
print('Accuracy: ', acc)


# In[23]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# In[24]:


from sklearn.tree import DecisionTreeClassifier


# In[25]:


# Create a decision tree classifier
clf = DecisionTreeClassifier()


# In[26]:


# Train the model on the training data
clf.fit(X_train, y_train)


# In[27]:


# Make predictions on the test data
y_pred = clf.predict(X_test)


# In[28]:


# Evaluate the model's performance
from sklearn.metrics import accuracy_score
print("Accuracy:", accuracy_score(y_test, y_pred))


# In[29]:


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[31]:


from sklearn.ensemble import RandomForestClassifier


# In[32]:


# Train the model
model = RandomForestClassifier()
model.fit(X_train, y_train)


# In[34]:


from sklearn.metrics import accuracy_score, precision_score, recall_score


# In[35]:


# evaluate the model
print("Accuracy: ", accuracy_score(y_test, y_pred))
print("Precision: ", precision_score(y_test, y_pred))
print("Recall: ", recall_score(y_test, y_pred))


# In[36]:


# Get the predicted probability
y_proba = log_reg.predict_proba(X_test)


# In[37]:


# Create a new column with the predicted probability
df_test = pd.DataFrame(X_test)
df_test['pred_proba'] = y_proba[:, 1]


# In[38]:


# Create a new column with the predicted likelihood
df_test['pred_likelihood'] = df_test['pred_proba'] * 100


# In[39]:


# Create a new column with the predicted outcome
df_test['pred_outcome'] = np.where(df_test['pred_likelihood']>50, 'high', 'low')


# In[40]:


# print the output 
print(df_test)


# In[41]:


print(df_test['pred_outcome'])


# In[42]:


data


# In[43]:


#save the model
joblib.dump(log_reg, 'log_reg_model.pkl')


# In[ ]:




