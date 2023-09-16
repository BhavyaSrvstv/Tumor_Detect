import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

data = load_breast_cancer()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

classifier = LogisticRegression()
classifier.fit(X_train, y_train)

new_tumor_data_malignant= np.array([[13.0, 21.82, 86.0, 527.2, 0.08, 0.19, 0.12, 0.09, 0.21, 0.07, 0.54, 1.39, 3.53, 48.0, 0.01, 0.03, 0.02, 0.02, 0.01, 0.005, 14.16, 24.04, 91.22, 633.5, 0.13, 0.4, 0.31, 0.2, 0.4, 0.1]])

new_tumor_data_benign = np.array([[12.0, 18.0, 100.0, 300.0, 0.1, 0.15, 0.12, 0.08, 0.18, 0.06, 0.45, 1.2, 2.8, 32.0, 0.008, 0.025, 0.015, 0.01, 0.015, 0.004, 13.0, 21.0, 110.0, 400.0, 0.15, 0.3, 0.2, 0.12, 0.25, 0.07]])

new_tumor_data_benign=scaler.transform(new_tumor_data_benign)

predicted_class = classifier.predict(new_tumor_data_benign)


class_probabilities = classifier.predict_proba(new_tumor_data_benign)[0]

plt.figure(figsize=(8, 6))
plt.scatter(['Malignant', 'Benign'], class_probabilities, c=['red', 'green'], s=100)
plt.xlabel('Tumor Class')
plt.ylabel('Probability')
plt.title('Tumor Classification Probability')
plt.ylim(0, 1)  
plt.grid(True)
plt.show()

# Print the predicted class label
if predicted_class == 0:
    print('The tumor is predicted to be Malignant.(Cancerous)')
else:
    print('The tumor is predicted to be Benign.(Non-cancerous)')
