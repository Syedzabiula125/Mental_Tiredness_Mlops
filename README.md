# 🧠 Mental Tiredness Score Prediction — MLOps Project

An end-to-end **Machine Learning + MLOps** project that predicts a mental tiredness score using personal and lifestyle-related inputs such as **age, sleep hours, study hours, and stress level**.

The project combines a **Random Forest Regression model** with a **Streamlit web application** and demonstrates practical MLOps concepts such as **model persistence, model versioning, validation metrics, prediction logging, and model metadata tracking**.

> ⚠️ **Disclaimer:** This project is an educational prototype. It is not a medical diagnostic system and should not be used as clinical advice or for real-world medical decision-making.

---

## 📌 Project Overview

Mental tiredness can be influenced by several lifestyle and behavioral factors. The objective of this project is to build an interactive application that takes a user's inputs and produces a **predicted tiredness score from 0 to 100**.

The application also converts the predicted score into three categories:

* **Low**
* **Moderate**
* **High**

The project focuses not only on model prediction, but also on how an ML model can be packaged, versioned, evaluated, monitored, and served through an application.

---

## 🎯 Problem Statement

Build a machine learning application that:

1. Accepts user-related input features.
2. Predicts a mental tiredness score.
3. Categorizes the prediction into Low, Moderate, or High.
4. Stores the trained model as an artifact.
5. Maintains model version information.
6. Tracks model evaluation metrics.
7. Logs every prediction for future monitoring and analysis.
8. Provides an interactive user interface using Streamlit.

---

## 🧩 Input Features

The model uses the following features:

| Feature        | Description                       |
| -------------- | --------------------------------- |
| `age`          | User age                          |
| `sleep_hours`  | Average sleep duration per night  |
| `study_hours`  | Study hours per day               |
| `stress_level` | Current stress level from 1 to 10 |

### Output

The application returns:

* Predicted tiredness score: **0–100**
* Tiredness category:

  * **Low**
  * **Moderate**
  * **High**

---

## 🤖 Machine Learning Model

The project uses a:

### Random Forest Regressor

`RandomForestRegressor` was selected as the starting algorithm because it can capture non-linear relationships in small tabular datasets without requiring extensive feature scaling.

The current implementation uses:

* `n_estimators = 250`
* `min_samples_leaf = 3`
* `random_state = 42`
* `n_jobs = -1`

The dataset is divided into training and testing portions using an **80/20 train-test split**.

---

## 📊 Model Evaluation

The model is evaluated using:

### Mean Absolute Error (MAE)

Measures the average absolute difference between the predicted and actual values.

### R² Score

Measures how well the model explains the variation in the target variable.

The validation metrics are stored as part of the model artifact and displayed in the Streamlit **MLOps Overview** section.

---

## ⚙️ MLOps Features

This project demonstrates several basic MLOps practices.

### 1. Model Persistence

The trained model and its metadata are stored using **Joblib**.

```text
app.joblib
```

If the model artifact is not available, the application generates a demonstration model and saves it.

### 2. Model Versioning

The application maintains an explicit model version:

```text
demo-rf-v1.0.0
```

This makes it possible to identify which model version produced a prediction.

### 3. Model Metadata

The stored model artifact contains information such as:

* Model version
* Algorithm name
* Feature names
* Creation timestamp
* Training source
* Validation metrics

### 4. Prediction Logging

Every prediction is logged locally into:

```text
prediction_logs.csv
```

The log contains information such as:

* Input features
* Prediction
* Prediction category
* Timestamp

This provides a foundation for future monitoring and analysis.

### 5. Streamlit Deployment

The trained ML model is integrated into a Streamlit application so users can interact with the model through a web interface.

---

## 🖥️ Application Structure

The Streamlit application contains two main sections:

### Prediction

Users enter:

```text
Age
Sleep Hours
Study Hours
Stress Level
```

The application then generates the predicted tiredness score and category.

### MLOps Overview

Displays:

* Model algorithm
* Validation MAE
* Validation R²
* Model version
* Model features
* Training source
* Model artifact path
* Prediction log path

---

## 🛠️ Tech Stack

### Programming Language

* Python

### Machine Learning

* Scikit-learn
* Random Forest Regressor

### Data Processing

* Pandas
* NumPy

### Model Persistence

* Joblib

### Application

* Streamlit

### Version Control

* Git
* GitHub

---

## 📁 Project Structure

```text
Mental_Tiredness_Mlops/
│
├── app.py
├── mental_tiredness_score_prediction_dataset.csv.csv
├── .gitignore
└── README.md
```

Generated files such as the model artifact and prediction logs are excluded through `.gitignore`.

---

## 🚀 Installation

### 1. Clone the repository

```bash
git clone https://github.com/Syedzabiula125/Mental_Tiredness_Mlops.git
```

### 2. Navigate to the project

```bash
cd Mental_Tiredness_Mlops
```

### 3. Create a virtual environment

Windows:

```bash
python -m venv project
```

### 4. Activate the environment

Windows PowerShell:

```powershell
.\project\Scripts\Activate.ps1
```

### 5. Install dependencies

Create a `requirements.txt` file with:

```text
streamlit
pandas
numpy
scikit-learn
joblib
```

Then install:

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Application

Start the Streamlit application using:

```bash
streamlit run app.py
```

The application will open in your browser.

---

## 🔄 Application Workflow

```text
User Input
    ↓
Feature Preparation
    ↓
Random Forest Regressor
    ↓
Predicted Tiredness Score
    ↓
Low / Moderate / High Category
    ↓
Prediction Logging
    ↓
Monitoring & Analysis
```

---

## 📌 Current Model Source

The current application contains a **synthetic demonstration-data generation step** when no trained model artifact is available.

The code creates a reproducible demonstration dataset and trains the Random Forest model before saving the model artifact.

For a production-quality system, this should be replaced with a **validated, consented, properly labelled real-world dataset**.

---

## 🔮 Future Improvements

Some possible improvements for future versions include:

* Use a validated real-world dataset.
* Add automated data validation.
* Compare Random Forest with Linear Regression and Gradient Boosting.
* Add cross-validation and hyperparameter tuning.
* Implement experiment tracking.
* Add automated model retraining.
* Add model monitoring dashboards.
* Add data drift and model drift detection.
* Create a CI/CD pipeline using GitHub Actions.
* Deploy the application to a cloud platform.
* Add automated testing.
* Containerize the application using Docker.

---

## 🎓 Learning Outcomes

Through this project, I gained practical experience in:

* Machine Learning model development
* Regression problems
* Model evaluation
* Streamlit application development
* Model serialization
* Model versioning
* Prediction logging
* Basic MLOps workflows
* Git and GitHub version control

---

## 👨‍💻 Author

**Syed Zabi**

Machine Learning & MLOps Learner

**Training Institute:** Innomatics Research Labs

---

## ⭐ Acknowledgement

Thanks to **Innomatics Research Labs** for providing the learning environment and guidance that supported the development of this project.

---

## 📄 License

This project is intended for educational and learning purposes.
