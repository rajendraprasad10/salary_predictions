import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

data = pd.read_csv('Salary_Data.csv')



x = data.iloc[:, :-1].values
y = data.iloc[:, 1].values


#Splitting Training and Test Set
#Since we have a very small dataset, we will train our model with all availabe data.

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()

#Fitting model with trainig data
regressor.fit(x, y)

# Saving model to disk
pickle.dump(regressor, open('model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
print(model.predict([[2]]))