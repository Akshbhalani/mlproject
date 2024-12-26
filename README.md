# Regression Analysis for Student Score Prediction

## Project Overview

This project is designed to predict students' math scores based on various factors such as gender, ethnicity, parental education level, lunch type, and test preparation course. It aims to explore how these features influence academic performance. The model is built using multiple machine learning algorithms to find the best prediction approach, and among all models, Linear Regression achieving the highest accuracy of 87% on training data. Additionally, the project also include a simple Flask web application that allows users to input student details and get predicted math scores.


## Key Features

+ **Machine Learning Models:** Implemented RandomForestRegressor, DecisionTreeRegressor, LinearRegression, GradientBoostingRegressor, and XGBRegressor to determine the best-performing model.
+ **Data Transformation:** Used StandardScaler for numerical data and OneHotEncoder for categorical data for effective data preprocessing.
+ **Web Application:** Built using Flask, enabling users to input details and receive predictions in real time.

## Exploratory Data Analysis (EDA)

  + Visualized and analyzed the dataset to uncover key insights, such as the relationship between different features and math scores.
  + Compared male and female performance to understand gender-based differences in scores.
 
 ## How It Works
+ **1. Data Injestion :**
  + Collect the data, split the data into train and test set.
+ **2. Data Preprocessing :**
    + Handled missing values and cleaned the dataset.
    + Scaled numerical columns using StandardScaler.
    + Encoded categorical columns with OneHotEncoder.
+ **3. Model Training :**
  + Trained multiple regression models and evaluated their performance.
  + Selected Linear Regression as the best-performing model with 87% accuracy.

+ **4. Web Application :**
  + Built a Flask app where users can input features like gender, ethnicity, parental education, lunch type, and test preparation course to predict math scores.
 
  ## Results

+  The Linear Regression model outperformed others with an accuracy of 87% on the training dataset.
+  Insights from EDA showed female performance is better than male performance. Also display significant relationships between features and students' math scores.

 
