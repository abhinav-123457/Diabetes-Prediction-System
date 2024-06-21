# Diabetes Prediction System

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-0.24.2-orange)
![Flask](https://img.shields.io/badge/Flask-2.0.1-green)

This project aims to predict the likelihood of diabetes in patients based on various health parameters using machine learning algorithms.

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
This project is licensed under the MIT License. See the LICENSE file for details.


### Notes:

- **Badges**: The badge URLs (`https://img.shields.io/badge/...`) should be updated with actual versions or statuses of the technologies used in your project.
- **Links**: The dataset link is provided as an example. Replace it with the actual link to your dataset if different.
- **Sections**: Each section (Overview, Dataset, Features, etc.) is structured with headings (`##`), and links within the document are formatted using markdown syntax `[text](URL)`.

This README.md template provides a structured and informative format to showcase your project on GitHub, ensuring it is clear and accessible to potential users and contributors. Adjust the content and sections based on your project's specific details and requirements.
