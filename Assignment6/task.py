import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier

bank_data = pd.read_csv('bank.csv', delimiter=';')
print(bank_data.head())
print(bank_data.dtypes)

selected_cols = bank_data[['y', 'job', 'marital', 'default', 'housing', 'poutcome']]
print(selected_cols.head())

encoded_data = pd.get_dummies(selected_cols, columns=['job', 'marital', 'default', 'housing', 'poutcome'])
print(encoded_data.head())

if encoded_data['y'].dtype == 'object':
    encoded_data['y'] = encoded_data['y'].map({'no': 0, 'yes': 1})

plt.figure(figsize=(14, 10))
sns.heatmap(encoded_data.corr(), annot=True, cmap='coolwarm')
plt.title("Feature Correlation Heatmap")
plt.show()

target_corr = encoded_data.corr()['y'].abs().sort_values(ascending=False)
print("Top 5 variables most correlated with target (by absolute value):")
print(target_corr.head(5))

"""
Top 5 variables most correlated with target (by absolute value):
y                   1.000000
poutcome_success    0.283481
poutcome_unknown    0.162038
housing_yes         0.104683
housing_no          0.104683
"""

target = encoded_data['y']
features = encoded_data.drop('y', axis=1)

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.25, random_state=42)

print("Logistic Regression:")
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
y_pred_log = log_reg.predict(X_test)

print("Classification Report:")
print(metrics.classification_report(y_test, y_pred_log))

conf_mat_log = metrics.confusion_matrix(y_test, y_pred_log)
print(f"Confusion Matrix:\n{conf_mat_log}")

acc_log = metrics.accuracy_score(y_test, y_pred_log)
print(f"\nAccuracy: {acc_log:.4f}")

metrics.ConfusionMatrixDisplay.from_estimator(log_reg, X_test, y_test)
plt.title("Logistic Regression Confusion Matrix")
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

print("K-Nearest Neighbors (K=3):")
knn_model = KNeighborsClassifier(n_neighbors=3)
knn_model.fit(X_train, y_train)
y_pred_knn = knn_model.predict(X_test)

print("Classification Report:")
print(metrics.classification_report(y_test, y_pred_knn))

conf_mat_knn = metrics.confusion_matrix(y_test, y_pred_knn)
print(f"\nConfusion Matrix:\n{conf_mat_knn}")

acc_knn = metrics.accuracy_score(y_test, y_pred_knn)
print(f"\nAccuracy: {acc_knn:.4f}")

metrics.ConfusionMatrixDisplay.from_estimator(knn_model, X_test, y_test)
plt.title("KNN Confusion Matrix")
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

"""
Findings:
    Logistic Regression achieved about 90% accuracy, while the K-Nearest Neighbors (K=3) model reached around 86%. 
    This shows that Logistic Regression performed slightly better in predicting the target variable. 
    The higher accuracy suggests that the relationship between the features and the target is mostly linear, 
    which suits Logistic Regression well. 
    KNN’s lower performance might be due to the use of categorical dummy variables or the small value of K, 
    which can make it sensitive to noise in the data. 
    Logistic Regression also offers better interpretability by showing how each feature influences the outcome. 
    Overall, Logistic Regression is a more reliable and efficient model for this dataset compared to KNN.
"""
