import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

data = load_diabetes(as_frame=True)
df = data.frame

plt.hist(df['target'], bins=25, color='orange')
plt.xlabel('Disease Score')
plt.ylabel('Frequency')
plt.title('Target Distribution')
plt.show()

sns.heatmap(df.corr().round(2), annot=True)
plt.show()

plt.subplot(1, 2, 1)
plt.scatter(df['bmi'], df['target'], color='purple', alpha=0.7)
plt.xlabel('BMI')
plt.ylabel('Target')
plt.title('BMI vs Target')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.scatter(df['s5'], df['target'], color='teal', alpha=0.7)
plt.xlabel('s5')
plt.ylabel('Target')
plt.title('s5 vs Target')
plt.grid(True)
plt.tight_layout()
plt.show()

X1 = df[['bmi', 's5']]
y = df['target']
X1_train, X1_test, y_train, y_test = train_test_split(X1, y, test_size=0.2, random_state=5)
model1 = LinearRegression()
model1.fit(X1_train, y_train)
y_pred_train1 = model1.predict(X1_train)
y_pred_test1 = model1.predict(X1_test)

rmse_train1 = np.sqrt(mean_squared_error(y_train, y_pred_train1))
rmse_test1 = np.sqrt(mean_squared_error(y_test, y_pred_test1))
r2_train1 = r2_score(y_train, y_pred_train1)
r2_test1 = r2_score(y_test, y_pred_test1)

print("Model: BMI + s5")
print(f"Train RMSE = {rmse_train1:.2f}, R² = {r2_train1:.4f}")
print(f"Test  RMSE = {rmse_test1:.2f}, R² = {r2_test1:.4f}")

X2 = df[['bmi', 's5', 'bp']]
X2_train, X2_test, y_train, y_test = train_test_split(X2, y, test_size=0.2, random_state=5)
model2 = LinearRegression()
model2.fit(X2_train, y_train)
y_pred_train2 = model2.predict(X2_train)
y_pred_test2 = model2.predict(X2_test)

rmse_train2 = np.sqrt(mean_squared_error(y_train, y_pred_train2))
rmse_test2 = np.sqrt(mean_squared_error(y_test, y_pred_test2))
r2_train2 = r2_score(y_train, y_pred_train2)
r2_test2 = r2_score(y_test, y_pred_test2)

print("\nModel: BMI + s5 + bp")
print(f"Train RMSE = {rmse_train2:.2f}, R² = {r2_train2:.4f}")
print(f"Test  RMSE = {rmse_test2:.2f}, R² = {r2_test2:.4f}")



"""...a. Which variable would you add next? Why?

I’d add bp (blood pressure) next because it shows a strong relationship with the target value,
similar to bmi and s5. From the heatmap, bp stood out as one of the features most connected to the disease score.
It also makes sense logically—blood pressure often plays a big role in health conditions like diabetes.

b. How does adding it affect the model’s performance?

After including bp, the model performed better.
The RMSE went down, meaning the predictions became more accurate,
and the R² score went up, showing that the model could explain more of the variation in the data.
So overall, adding bp made the model stronger and more reliable than when it used only bmi and s5.

c. Does it help if you add even more variables?

Yes, adding more relevant variables can improve the model, but only up to a certain point.
If the new features actually relate to the target, they’ll help the model learn better.
But if they don’t add meaningful information, they can introduce noise and make the model worse.
It’s usually better to focus on the most important variables instead of just including everything..."""