import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np


data = pd.read_csv('weight-height.csv')


X = data[["Height"]].values
y = data["Weight"].values


plt.scatter(X, y, color='red')
plt.xlabel('Height')
plt.ylabel('Weight')
plt.title('Scatter Plot of Weight vs Height')
plt.show()


lin_reg = LinearRegression()
lin_reg.fit(X, y)

y_pred = lin_reg.predict(X)


plt.scatter(X, y, color='red')
plt.plot(X, y_pred, color='blue', label='Regression Line')
plt.xlabel('Height')
plt.ylabel('Weight')
plt.title('Linear Regression of Weight vs Height')
plt.legend()
plt.show()


rmse = np.sqrt(mean_squared_error(y, y_pred))
r2 = r2_score(y, y_pred)

print("Root Mean Squared Error =", rmse)
print("R² =", r2)
#guri
