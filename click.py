import pandas as pd
import time


import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


#path = 'C:/Users/mariusz/Documents/qg/adclick/train.csv'
#path1 = 'C:/Users/mariusz/Documents/qg/adclick/test.csv'

path = '/home/mariusz/adclick/train.csv'
path1 = '/home/mariusz/adclick/test.csv'
# start = time.clock()

clicks = pd.read_csv(path,nrows=5e6)
clicks_test = pd.read_csv(path1)

# end = time.clock()
# print(end - start)
# print(sum(df.is_attributed))

from sklearn.metrics import accuracy_score

train_X, test_X, train_y, test_y = train_test_split(clicks.iloc[:,:5],pd.DataFrame(clicks.is_attributed),test_size= 0.3, random_state=17278)

#clicks.describe()

# Create linear regression object
regr = linear_model.LogisticRegression(class_weight='balanced')


# Train the model using the training sets
regr.fit(train_X, train_y.values.ravel())

# Make predictions using the testing set
train_y_pred = regr.predict(train_X)
test_y_pred = regr.predict(test_X)

print(accuracy_score(train_y, train_y_pred))
print(accuracy_score(test_y, test_y_pred))





