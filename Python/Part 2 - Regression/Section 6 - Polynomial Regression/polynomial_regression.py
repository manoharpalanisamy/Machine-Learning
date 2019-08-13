# Polynomial Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

# There is not enough data we have , we comment Train $ test
# Splitting the dataset into the Training set and Test set
"""from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)"""

# Feature Scaling (Liraries will take care no need to do Feature Scaling)
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)"""

# Fitting Linear Regression to the dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)

# Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 5) # added x0=1 automatically and create 5 Independent variables
X_poly = poly_reg.fit_transform(X)   # line 30 to 32 we tranform X to X_poly
lin_reg_2 = LinearRegression()     
lin_reg_2.fit(X_poly, y)             # line 33, 34 we perform LinearRegression with Ploynomial terms

# Visualising Linear Regression Results
plt.scatter(X, y, color = 'Red')
plt.plot(X, lin_reg.predict(X), color = 'blue') # Plot Prediction (Straight Line)
plt.title('Truth or Bluff (Linear Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# Visualising Polynomial Regression Results
X_grid = np.arange(min(X), max(X), 0.1)               # line 45, 46 smoothen curve 
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'Red')
plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)), color = 'blue') # Plot Prediction for lin_reg_2(curved function)create 5 Independent variables
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

#Predicting new results with Linear Regression
lin_reg.predict([[6.5]]) # Predict level 6.5

#Predicting new results with Polynomial Regression
lin_reg_2.predict(poly_reg.fit_transform([[6.5]]))
