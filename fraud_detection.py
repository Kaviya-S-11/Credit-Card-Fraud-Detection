# ===============================
#   CREDIT CARD FRAUD DETECTION
# ===============================

#--Import required libraries--
import numpy as np               # For numerical operations
import pandas as pd              # For data handling (CSV files)
import matplotlib.pyplot as plt  # For plotting graphs
import seaborn as sns            # For advanced data visualization

# Machine Learning tools from scikit-learn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# -- Load the dataset --
data = pd.read_csv("creditcard.csv")  # read the CSV file

# Display first 5 rows of the dataset
print("First 5 rows of the dataset:")
print(data.head())

# Display number of rows and  columnns
print("\nDataset shape (row, columns):")
print(data.shape)

# -- Target Column --
# count normal (0) and fraud (1) transactions
print("\nCount of normal vs fraud transactions:")
print(data['Class'].value_counts())

# -- Visualize fraud vs normal transactions --
sns.countplot(x='Class', data=data)
plt.title("Fraud vs Normal Transactions")
plt.xlabel("Class (0 = Normal, 1 = Fraud)")
plt.ylabel("Number of Transactions")
plt.show()

# -- Separate features and target --
# X contains all columns except 'Class'
X = data.drop('Class', axis=1)

#y contains only the 'class' column (target)
y = data['Class']

# --Split the data into training and testing sets --

# 80% data for training and 20% for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, 
                                                    random_state = 42)

# -- Train Logistics Regression model --
# create the model
model = LogisticRegression(max_iter=1000)

#Train the model using training data
model.fit(X_train, y_train)

# --Make Predictions --
# Predict results on test data
y_pred = model.predict(X_test)

# --Evaluating the model --

#Accuracy score
print("\nModel Accuracy:")
print(accuracy_score(y_test, y_pred))

# Confusion Matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

#Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# --Handle Imbalanced Dataset --
# separate normal and fraud transactions
normal = data[data.Class == 0]
fraud = data[data.Class == 1]

#Random sample
normal_sample = normal.sample(n=len(fraud), random_state = 42)

#Combine fraud and sample normal data
balanced_data = pd.concat([normal_sample, fraud])

# Shuffle the balanced dataset
balanced_data = balanced_data.sample(frac=1, random_state=42)

print("\Blanced dataset class count:")
print(balanced_data['Class'].value_counts())

# -- Repeat traing with balanced data --

X_balanced = balanced_data.drop('Class', axis=1)
y_balanced = balanced_data['Class']

X_train, X_test, y_train, y_test = train_test_split(X_balanced, y_balanced, test_size = 0.2, random_state = 42)

#Train model again
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predictions 
y_pred = model.predict(X_test)

# -- Evaluate balanced model --
print("\nBalanced Model Accuracy:")
print(accuracy_score(y_test, y_pred))

print("\nBalanced Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nBalanced Classification Report:")
print(classification_report(y_test, y_pred))

print("\nCredit Card Fraud Detection Completed Successfully")