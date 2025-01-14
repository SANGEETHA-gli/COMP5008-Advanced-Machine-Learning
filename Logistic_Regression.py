import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
)
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


data = pd.read_csv(r"C:\Users\dsang\source\repos\Machine_Learning_Assignment\Dataset\preprocessed_data.csv")


X = data.iloc[:, :-1]
y = data['Outcome']


best_k = None
best_accuracy = 0
print("Finding best k for Logistic Regression...")

for k in range(2, 11):
    model = LogisticRegression(max_iter=1000, random_state=42)
    scores = cross_val_score(model, X, y, cv=k, scoring='accuracy')
    mean_accuracy = np.mean(scores)
    print(f"Logistic Regression - k={k}, Accuracy: {mean_accuracy:.4f}")
    if mean_accuracy > best_accuracy:
        best_accuracy = mean_accuracy
        best_k = k

print(f"\nBest k for Logistic Regression: {best_k}, Best Accuracy: {best_accuracy:.4f}")


stratified_kfold = StratifiedKFold(n_splits=best_k, shuffle=True, random_state=42)
model = LogisticRegression(max_iter=1000, random_state=42)


y_pred_cv = cross_val_predict(model, X, y, cv=stratified_kfold, method="predict")
y_proba_cv = cross_val_predict(model, X, y, cv=stratified_kfold, method="predict_proba")[:, 1]


accuracy = accuracy_score(y, y_pred_cv)
precision = precision_score(y, y_pred_cv)
recall = recall_score(y, y_pred_cv)
f1 = f1_score(y, y_pred_cv)
roc_auc = roc_auc_score(y, y_proba_cv)

print(f"\nLogistic Regression Metrics (k={best_k}):")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")
print(f"ROC-AUC: {roc_auc:.4f}")


print("\nClassification Report:")
print(classification_report(y, y_pred_cv))


conf_matrix = confusion_matrix(y, y_pred_cv)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title(f"Confusion Matrix (k={best_k})")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


fpr, tpr, thresholds = roc_curve(y, y_proba_cv)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'Logistic Regression (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random Guess')
plt.title(f"Receiver Operating Characteristic (ROC) Curve (k={best_k})")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower right")
plt.grid()
plt.show()
