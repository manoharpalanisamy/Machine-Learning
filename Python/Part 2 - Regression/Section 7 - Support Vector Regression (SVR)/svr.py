# SVR (we need to apply feature scaling)

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, 2:].values

# There is not enough data we have 10 eg. , we comment Train $ test
# Splitting the dataset into the Training set and Test set
"""from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)"""

# Feature Scaling (Liraries will take care no need to do Feature Scaling)
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
# sc_X.fit() # Compute the mean and std to be used for later scaling.
X = sc_X.fit_transform(X)  # Fit to data, then transform it
y = sc_y.fit_transform(y)

# Fitting the SVR to the dataset
from sklearn.svm import SVR    # (SVR doenst have , we need to apply feature scaling)
# Create your regressor here
regressor = SVR(kernel = 'rbf')   # rbf means guassian ((non-linear))
regressor.fit(X, y)

#Predicting new results with SVR
y_pred = sc_y.inverse_transform(regressor.predict(sc_X.transform(np.array([[6.5]]))))   # object sc_X already fitted , we do transform() needs array So, np.array()
# inverse_transform does get back our original values from the scaled one

# Visualising SVR Results
plt.scatter(X, y, color = 'Red')
plt.plot(X, regressor.predict(X), color = 'blue') 
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# Visualising SVR Results (for Higher resolution and smoother curve)
X_grid = np.arange(min(X), max(X), 0.1) # gives vector               # line 45, 46 smoothen curve 
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'Red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue') # plot a curve 
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

