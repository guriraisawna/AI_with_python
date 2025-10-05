import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

data = pd.read_csv("50_Startups.csv")
print(data.head())
print(data.info())

num_data = data.select_dtypes(include=[np.number])
print(num_data.corr())

plt.figure(figsize=(8, 6))
sns.heatmap(num_data.corr().round(2), annot=True, cmap='Blues')
plt.title('Correlation Heatmap')
plt.show()

data = pd.get_dummies(data, columns=['State'], drop_first=True)
X = data.drop('Profit', axis=1)
y = data['Profit']

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.scatter(data['R&D Spend'], data['Profit'])
plt.xlabel('R&D Spend')
plt.ylabel('Profit')
plt.title('Profit vs R&D Spend')

plt.subplot(1, 2, 2)
plt.scatter(data['Marketing Spend'], data['Profit'], color='green')
plt.xlabel('Marketing Spend')
plt.ylabel('Profit')
plt.title('Profit vs Marketing Spend')

plt.tight_layout()
plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

print(f"Train RMSE = {train_rmse:.2f}, R² = {train_r2:.4f}")
print(f"Test  RMSE = {test_rmse:.2f}, R² = {test_r2:.4f}")


""" findings  :

The dataset contains 50 rows and includes R&D Spend, 
Administration, Marketing Spend, State, and Profit. 
Among these, R&D Spend shows the strongest connection with Profit, 
followed by Marketing Spend, while Administration doesn’t seem to have much impact.
The scatter plots clearly show that Profit tends to increase with higher R&D and Marketing Spend. 
The linear regression model performs well, with high R² and low RMSE on both the training and testing data. 
Overall, R&D Spend appears to be the most important factor for predicting Profit."""