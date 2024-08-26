import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pickle

# Load the dataset
data = pd.read_csv('CVD_cleaned.csv')

# Print unique values in each column for inspection
for column in data.columns:
    unique_vals = data[column].unique()
    print(f"Unique values in column '{column}': {unique_vals}")

# Data Preprocessing
# Mapping categorical variables to numerical values
health_map = {'Excellent': 0, 'Very Good': 1, 'Good': 2, 'Fair': 3, 'Poor': 4}
data['General_Health'] = data['General_Health'].map(health_map)

binary_map = {'Yes': 1, 'No': 0}
data['Heart_Disease'] = data['Heart_Disease'].replace(binary_map)
data['Other_Cancer'] = data['Other_Cancer'].replace(binary_map)
data['Depression'] = data['Depression'].replace(binary_map)
data['Skin_Cancer'] = data['Skin_Cancer'].replace(binary_map)
data['Arthritis'] = data['Arthritis'].replace(binary_map)
data['Smoking_History'] = data['Smoking_History'].replace(binary_map)
data['Sex'] = data['Sex'].replace({'Female': 1, 'Male': 0})

# Label encoding for categorical features
encoder = LabelEncoder()
data['Age_Category'] = encoder.fit_transform(data['Age_Category'])
data['Diabetes'] = encoder.fit_transform(data['Diabetes'])
data['Checkup'] = encoder.fit_transform(data['Checkup'])
data['Exercise'] = encoder.fit_transform(data['Exercise'])

# Define features and target
X = data.drop(columns=['Heart_Disease'])
y = data['Heart_Disease']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Standardize the feature variables
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train a logistic regression model
lr = LogisticRegression(random_state=0)
lr.fit(X_train_scaled, y_train)

# Make predictions on the test set
y_pred = lr.predict(X_test_scaled)

# Evaluate the model
print('Accuracy score:', accuracy_score(y_test, y_pred))
print('Precision score:', precision_score(y_test, y_pred))
print('Recall score:', recall_score(y_test, y_pred))
print('F1 score:', f1_score(y_test, y_pred))

# Save the trained model and the scaler as pickle files
pickle.dump(lr, open('logistics.pkl', 'wb'))
pickle.dump(scaler, open('scaler.pkl', 'wb'))
