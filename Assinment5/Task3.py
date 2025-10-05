import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import r2_score

df = pd.read_csv("Auto.csv")
df = df.dropna()

X = df.drop(columns=["mpg", "name", "origin"])
y = df["mpg"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

alphas = [0.1, 0.2, 0.3, 1, 2, 3, 5, 10]
ridge_r2 = []
lasso_r2 = []

for alpha in alphas:
    ridge_model = Ridge(alpha=alpha)
    ridge_model.fit(X_train, y_train)
    ridge_r2.append(r2_score(y_test, ridge_model.predict(X_test)))

    lasso_model = Lasso(alpha=alpha, max_iter=5000)
    lasso_model.fit(X_train, y_train)
    lasso_r2.append(r2_score(y_test, lasso_model.predict(X_test)))

plt.plot(alphas, ridge_r2, marker='o', label="Ridge")
plt.plot(alphas, lasso_r2, marker='o', label="LASSO")
plt.xlabel("Alpha Value")
plt.ylabel("R² Score")
plt.title("Ridge vs LASSO Performance")
plt.legend()
plt.show()



"""  Findings:

I used Ridge and LASSO regression to predict car MPG using all the numeric features, excluding mpg, name, and origin. 
I tested both models with different alpha values to see how they performed. 
For Ridge, the best R² score was about 0.79 at alpha = 0.1, and LASSO gave a similar score at the same alpha.
In general, smaller alpha values worked better, while higher alphas made the models underfit and perform worse.
Overall, Ridge did a little better, but both models gave good results for predicting MPG."""