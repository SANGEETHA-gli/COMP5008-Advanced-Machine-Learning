# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.naive_bayes import GaussianNB
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
import seaborn as sns
import matplotlib.pyplot as plt


data = pd.read_csv(r"C:\Users\dsang\source\repos\Machine_Learning_Assignment\Dataset\preprocessed_data.csv")
X = data.iloc[:, :-1]
y = data['Outcome']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

best_k = None
best_accuracy = 0
print("Finding best k for Naïve Bayes...")

for k in range(2, 11):
    model = GaussianNB()
    scores = cross_val_score(model, X_train, y_train, cv=k, scoring='accuracy')
    mean_accuracy = np.mean(scores)
    print(f"Naïve Bayes - k={k}, Accuracy: {mean_accuracy:.4f}")
    if mean_accuracy > best_accuracy:
        best_accuracy = mean_accuracy
        best_k = k

print(f"\nBest k for Naïve Bayes: {best_k}, Best Accuracy: {best_accuracy:.4f}")


stratified_kfold = StratifiedKFold(n_splits=best_k, shuffle=True, random_state=42)
model = GaussianNB()


y_pred_cv_train = cross_val_predict(model, X_train, y_train, cv=stratified_kfold, method="predict")
y_proba_cv_train = cross_val_predict(model, X_train, y_train, cv=stratified_kfold, method="predict_proba")[:, 1]


model.fit(X_train, y_train)
y_pred_test = model.predict(X_test)
y_proba_test = model.predict_proba(X_test)[:, 1]


accuracy = accuracy_score(y_test, y_pred_test)
precision = precision_score(y_test, y_pred_test)
recall = recall_score(y_test, y_pred_test)
f1 = f1_score(y_test, y_pred_test)
roc_auc = roc_auc_score(y_test, y_proba_test)

print(f"\nNaïve Bayes Metrics (Test Data):")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")
print(f"ROC-AUC: {roc_auc:.4f}")


print("\nClassification Report (Test Data):")
print(classification_report(y_test, y_pred_test))

conf_matrix = confusion_matrix(y_test, y_pred_test)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix (Test Data)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

fpr, tpr, thresholds = roc_curve(y_test, y_proba_test)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'Naïve Bayes (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random Guess')
plt.title("Receiver Operating Characteristic (ROC) Curve (Test Data)")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower right")
plt.grid()
plt.show()
