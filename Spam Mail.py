
# Importing the Libraries

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score


# In[2]:


# Data Collection and Preprocessing

raw_mail_data = pd.read_csv('mail_data.csv')


print(raw_mail_data)

# Replace the null values with a null string

mail_data = raw_mail_data.where((pd.notnull(raw_mail_data)), '')

mail_data.head()

mail_data.shape


# Label Encoding

mail_data.loc[mail_data['Category'] == 'spam', 'Category',] = 0
mail_data.loc[mail_data['Category'] == 'ham', 'Category',] = 1


# Separating the data as texts and label

X = mail_data['Message']
y = mail_data['Category']

print(X)
print(y)


# Splitting the data into train and test data

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 3)

print(X.shape, X_train.shape, X_test.shape)


# Convert the text data in "Message" column to feature vectors using feature extraction

feature_extraction = TfidfVectorizer(min_df = 1, stop_words='english', lowercase = 'True')

X_train_features = feature_extraction.fit_transform(X_train)
X_test_features = feature_extraction.transform(X_test)

# Convert y_train and y_test as integers

y_train = y_train.astype('int')
y_test = y_test.astype('int')


print(X_train)
print(X_train_features)


# Model Training

model = LogisticRegression()

model.fit(X_train_features, y_train)


# Evaluation of trained model

prediction_on_training_data = model.predict(X_train_features)
accuracy_on_training_data = accuracy_score(y_train, prediction_on_training_data)

print('Accuracy on training data : ', accuracy_on_training_data)

# Evaluation of test model

prediction_on_test_data = model.predict(X_test_features)
accuracy_on_test_data = accuracy_score(y_test, prediction_on_test_data)

print('Accuracy on test data : ', accuracy_on_test_data)


# Building a Predictive system

input_mail = ["Nah I don't think he goes to usf, he lives around here though"]

# Convert text to feature vectors
input_data_features = feature_extraction.transform(input_mail)

# Making prediction

prediction = model.predict(input_data_features)
print(prediction)

if (prediction[0] == 1):
    print('This is a Ham mail')
else:
    print('This is a Spam mail')
