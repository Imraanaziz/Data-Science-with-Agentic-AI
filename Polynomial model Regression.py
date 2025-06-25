import pandas as pd
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt 

from sklearn.preprocessing import PolynomialFeatures

dataset = pd.read_csv(r"E:\Download\emp_sal.csv") 

x = dataset.iloc[:, 1:2].values 
y = dataset.iloc[:, 2].values 

from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(x, y)

plt.scatter(x, y, color = 'red')
plt.plot(x, lin_reg.predict(x), color = 'blue')
plt.title('linear regression model (linear Regression)')
plt.xlabel('position level')
plt.ylabel('salary')
plt.show() 

lin_model_pred = lin_reg.predict([[6.5]])
lin_model_pred 

#Non Linear Model

poly_reg = PolynomialFeatures(degree = 6)
x_poly = poly_reg.fit_transform(x)  

poly_reg.fit(x_poly, y)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(x_poly, y) 

#1st model lin_reg_2 (linear model)
#2nd model poly_reg(Polynomial model 

plt.scatter(x, y, color = 'red')
plt.plot(x, lin_reg_2.predict(poly_reg.fit_transform(x)), color = 'blue')
plt.title('polynomial Regression model (polynomial Regression)')
plt.xlabel('position level')
plt.ylabel('salary')
plt.show()  

poly_model_pred = lin_reg_2.predict(poly_reg.fit_transform([[6.5]]))
poly_model_pred