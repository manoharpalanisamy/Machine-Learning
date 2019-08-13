# Polynomial Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

# There is not enough data we have 10 eg. , we comment Train $ test
# Splitting the dataset into the Training set and Test set
"""from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)"""

# Feature Scaling (Liraries will take care no need to do Feature Scaling)
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)"""

# Fitting the Regression Model to the dataset
# Create your regressor here

#Predicting new results with Polynomial Regression
y_pred = regressor.predict([[6.5]])


# Visualising Polynomial Regression Results
plt.scatter(X, y, color = 'Red')
plt.plot(X, regressor.predict(X), color = 'blue') # Plot Prediction for lin_reg_2(curved function)create 5 Independent variables
plt.title('Truth or Bluff (Regression Model)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# Visualising Polynomial Regression Results (for Higher resolution and smoother curve)
X_grid = np.arange(min(X), max(X), 0.1) # gives vector               # line 45, 46 smoothen curve 
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'Red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue') # Plot Prediction for lin_reg_2(curved function)create 5 Independent variables
plt.title('Truth or Bluff (Regression Model)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()


