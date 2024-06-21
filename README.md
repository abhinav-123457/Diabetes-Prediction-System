# Diabetes Prediction System

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-0.24.2-orange)
![Flask](https://img.shields.io/badge/Flask-2.0.1-green)

This project aims to predict the likelihood of diabetes in patients based on various health parameters using machine learning algorithms.
![ProjectImage](webpage(1).jpg)
## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Features](#features)
- [Models](#models)
- [Deployment](#deployment)
- [Usage](#usage)
- [Setup](#setup)
- [Contributing](#contributing)
- [License](#license)

## Overview

Diabetes is a prevalent health condition affecting millions worldwide. Early detection and prediction can significantly improve patient outcomes. This project leverages machine learning techniques to predict diabetes based on patient health metrics.

## Dataset

The dataset used in this project is the [Pima Indians Diabetes Database](https://www.kaggle.com/uciml/pima-indians-diabetes-database), which is publicly available on Kaggle. It consists of various attributes such as Glucose level, BMI, Age, Blood Pressure, etc.

## Features

The features used for predicting diabetes include:
- Glucose
- BMI (Body Mass Index)
- Age
- Blood Pressure
- Insulin levels
- Diabetes Pedigree Function
- Skin Thickness
- Pregnancies (for females)

These features are crucial indicators in diabetes diagnosis and management.

## Models

Several machine learning models were evaluated for their performance in predicting diabetes, including:
- Logistic Regression
- Random Forest Classifier
- Support Vector Machine (SVM)
- Decision Tree Classifier

The models were trained and evaluated based on metrics such as accuracy, precision, recall, and F1-score. The best performing model was chosen for deployment.

## Code of the model: 

    !pip install pandas numpy scikit-learn joblib

    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer
    from sklearn.ensemble import RandomForestClassifier
    import joblib

    # Load dataset  
    data = pd.read_csv('/content/diabetes.csv')

    # Handle missing values
    imputer = SimpleImputer(strategy='mean')
    data.iloc[:, :-1] = imputer.fit_transform(data.iloc[:, :-1])

    # Split data into features and target 
    X = data.drop('Outcome', axis=1)
    y = data['Outcome']

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Normalize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Train the model
    model = RandomForestClassifier(n_estimators=100, max_depth=None, random_state=42)
    model.fit(X_train, y_train)
 
    # Save the model and scaler
    joblib.dump(model, 'best_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    from google.colab import files

    # Download the model
    files.download('best_model.pkl')

    # Download the scaler
    files.download('scaler.pkl')


## File stucture:

     
     diabetes_prediction/
    ├── app.py
    ├── best_model.pkl
    ├── scaler.pkl
    └── templates/
        └── index.html

## Deployment

The prediction model is deployed using a Flask web application. Users can input their health metrics, and the system will predict the likelihood of diabetes based on the trained model.

## Usage

To use this project locally, follow these steps:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/abhinav-123457/Diabetes-Prediction-System.git
   cd Diabetes-Prediction-System
   
2. **Install dependencies:**
   ```bash
   pip install flask numpy joblib
   
3. **Run the application:(Type in the terminal)**
   ```bash
   python app.py

4. **Click the link come in the terminal after running app.py**
 
## Setup

Ensure you have Python 3.8 or higher installed. Use pip to install required packages.

## Contributing
Contributions are welcome! If you'd like to add new features, improve the model, or fix issues, please fork the repository and create a pull request.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.


### Notes:

- **Badges**:  ![Python](https://img.shields.io/badge/Python-3.8%2B-blue)  ![scikit-learn](https://img.shields.io/badge/scikit--learn-0.24.2-orange)  ![Flask](https://img.shields.io/badge/Flask-2.0.1-green)
- **Sections**: Each section (Overview, Dataset, Features, etc.) is structured with headings (`##`), and links within the document are formatted using markdown syntax `[text](URL)`.

This README.md template provides a structured and informative format to showcase your project on GitHub, ensuring it is clear and accessible to potential users and contributors. Adjust the content and sections based on your project's specific details and requirements.
