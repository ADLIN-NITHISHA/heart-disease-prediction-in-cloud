💓 HEART DISEASE PREDICTION USING MACHINE LEARNING ALGORITHMS

🧠 Project Overview

This project aims to predict the likelihood of cardiovascular disease using patient health data. The objective is to assist in early diagnosis by applying various machine learning algorithms and deploying the model using Flask and AWS.

📊 Problem Statement

Cardiovascular diseases are one of the leading causes of death globally. Early prediction can help reduce risk and improve treatment. This project uses health-related features to build a binary classification model that predicts whether a person has heart disease or not.

🛠️ Tools & Technologies

Programming Language: Python
Libraries Used: pandas, numpy, matplotlib, seaborn, scikit-learn
Machine Learning Algorithms: Logistic Regression, Random Forest, K-Nearest Neighbors
Model Deployment: Flask Web App, AWS Hosting
Collaboration: Team of 3 students
Environment: Jupyter Notebook, colab

📁 Dataset

Source: UCI Heart Disease Dataset

Features:
Age, Sex, Chest Pain Type, Blood Pressure, Cholesterol, Fasting Blood Sugar, ECG, Heart Rate, Exercise Induced Angina, ST Depression, etc.
Target: Presence of heart disease (1 = Yes, 0 = No)

🔍 Exploratory Data Analysis (EDA)

Handled missing values and performed outlier detection
Visualized feature distributions using seaborn
Explored correlations and trends between variables
Balanced the dataset using SMOTE (if applicable)

🤖 Model Building

Trained and evaluated multiple models
Used accuracy, precision, recall, and confusion matrix for evaluation
Tuned hyperparameters using GridSearchCV

🚀 Deployment

Created a Flask web app with a simple UI to input patient data
Deployed the application using AWS EC2
Accessed the app via a public IP

📈 Results

Best model: Logistic Regression
Achieved an accuracy of ~85%
Provided real-time prediction via the web app

🤝 Contribution

This was a team project developed by three members during our Master’s program. My role involved:
Performing data cleaning and EDA
Building and testing machine learning models
Integrating the model with Flask for deployment


