# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 14:37:19 2019

@author: Raul Paz Kaggle Project
"""

# Kaggle Regression

# Import Libraries

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import metrics

# Read Data
train = pd.read_csv(r"C:\Users\Dustin\Desktop\Machine Learning\kaggle\house-prices-advanced-regression-techniques\train.csv")

#train.describe()
#test.describe()

# Assign IDs and fill null values
def processing(dat):
    
    ID = dat['Id']
    del dat['Id']
    
    dat = dat.apply(lambda x: x.fillna(0) if x.dtype.kind in 'biufc' else x.fillna('None'))
    
    return ID, dat

# Clean Data
Id_train,train = processing(train)

# Assign KPI of dataset
x_train = train.drop(columns = 'SalePrice')
y_train = train['SalePrice']

# Create Dummy variables and standardize test data
sc = StandardScaler()
x_train = pd.get_dummies(x_train)
a = np.array(x_train.columns.values)
sc.fit_transform(x_train)
x_train = sc.transform(x_train)
x_train = pd.DataFrame(data = x_train,columns = a)

# rescale cond and qual
x_train['OverallCond'] = (train['OverallCond']-5.5)/(4.5/3)
x_train['OverallQual'] = (train['OverallQual']-5.5)/(4.5/3)
#x_train['OverallQual'] = (train['OverallQual']-5.5)*(6/9) another way

#test-train split
X_train, X_test, Y_train, Y_test = train_test_split(x_train, y_train, test_size=0.25, random_state=42)

score = {}
#for i in ['auto', 'sqrt', 'log2']: #used on max_features
for i in [25,50,75,100]: #used on n_estimators and min_samples_split
    # Create Random forest Regression model
    regressor = RandomForestRegressor(max_features = 'auto', n_estimators=75, min_samples_split = 25, random_state=0)
    regressor.fit(X_train, Y_train)
    y_pred = regressor.predict(X_train)

    # Performance on training dataset
    print('\nMean Absolute Error:', metrics.mean_absolute_error(Y_train, y_pred))
    print('Mean Squared Error:', metrics.mean_squared_error(Y_train, y_pred))
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(Y_train, y_pred)))
    print('Score:', regressor.score(X_train, Y_train))
    train_score = regressor.score(X_train, Y_train)

    # Run prediction on test data
    Y_test_pred = regressor.predict(X_test)

    # Performance on test set
    print('\nMean Absolute Error:', metrics.mean_absolute_error(Y_test, Y_test_pred))
    print('Mean Squared Error:', metrics.mean_squared_error(Y_test, Y_test_pred))
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(Y_test, Y_test_pred)))
    print('Score:', regressor.score(X_test, Y_test))
    test_score = regressor.score(X_test, Y_test)
    
    #Create Dictionary
    score[i] = [train_score, test_score]

# Plot
df_score = pd.DataFrame(score, index = ['Train', 'Test']).T
df_score.plot(figsize = (12,8), style = 'o--')

#auto was best on both test and train for max_features
#75 trees was best for test on n_estimators (2nd best on train)
#25 was best on min # of observations allowed in leaf

#final Regression
regressor = RandomForestRegressor(max_features = 'auto', n_estimators=75, min_samples_split = 25, random_state=0)
regressor.fit(x_train, y_train)
y_pred = regressor.predict(x_train)

# Performance on training dataset
print('\nMean Absolute Error:', metrics.mean_absolute_error(y_train, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_train, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_train, y_pred)))
print('Score:', regressor.score(x_train, y_train))



###############################################################################
#Kaggle Test
###############################################################################

test = pd.read_csv(r"C:\Users\Dustin\Desktop\Machine Learning\kaggle\house-prices-advanced-regression-techniques\test.csv")
Id_test,test = processing(test)
test['OverallQual'].replace(quality, inplace = True)
test['OverallCond'].replace(quality, inplace = True)



# Run Random Forest on 
temp_test = pd.get_dummies(test)
x_test = pd.DataFrame(np.zeros((temp_test.shape[0],len(a))),columns = a)
x_test.update(temp_test)
x_test = pd.DataFrame(data = sc.transform(x_test),columns = a)

# rescale cond and qual
x_test['OverallCond'] = (test['OverallCond']-5.5)/(4.5/3)
x_test['OverallQual'] = (test['OverallQual']-5.5)/(4.5/3)

# Create Output
Out = pd.DataFrame(Id_test)
Out['SalePrice'] = regressor.predict(x_test)
Out['SalePrice'].describe()
#Out.to_csv('../PredictSellingPrice/submission.csv',index = False)


