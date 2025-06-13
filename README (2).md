
# ❤️ Heart Disease Prediction using Logistic Regression

This project demonstrates how to build a machine learning model to predict the presence of heart disease using a logistic regression algorithm. The entire workflow — from data loading to prediction — is implemented within a single Jupyter Notebook.

---

## 📁 Project Contents

```
heart-disease-prediction/
│
├── heart_disease_data.csv       # Dataset (UCI-based)
├── heart_disease_prediction.ipynb  # Jupyter Notebook with model training and prediction
└── README.md                    # Project documentation
```

---

## 🚀 How to Use

### 1. ✅ Requirements

Make sure Python is installed, then install the necessary libraries:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

### 2. 📓 Run the Notebook

Open the notebook with Jupyter:

```bash
jupyter notebook heart_disease_prediction.ipynb
```

Follow the notebook steps to:

- Load and explore the dataset
- Preprocess the data (handling missing values, encoding)
- Train a **Logistic Regression** model
- Evaluate performance with accuracy, precision, recall
- Test the model with custom input values

---

## 🧪 Sample Prediction

The notebook includes an example like:

```python
input_data = (62, 0, 0, 140, 268, 0, 0, 160, 0, 3.6, 0, 2, 2)
input_array = np.array(input_data).reshape(1, -1)
prediction = model.predict(input_array)

if prediction[0] == 0:
    print("No Heart Disease")
else:
    print("Patient has Heart Disease")
```

---

## 📊 Dataset Features

The dataset includes 13 input features:

1. age
2. sex
3. cp (chest pain type)
4. trestbps (resting blood pressure)
5. chol (cholesterol level)
6. fbs (fasting blood sugar > 120 mg/dl)
7. restecg (resting ECG)
8. thalach (max heart rate)
9. exang (exercise-induced angina)
10. oldpeak (ST depression)
11. slope (slope of the peak exercise ST segment)
12. ca (number of major vessels)
13. thal (thalassemia)

Target variable: `0` = No Heart Disease, `1` = Heart Disease

---

## ✅ Model Summary

- **Model Used**: Logistic Regression
- **Evaluation Metrics**: Accuracy, Confusion Matrix, Precision, Recall
- **Approach**: All steps are done within one notebook, no external files used

---

## 🙋‍♂️ Author

Developed by Kathirvel Kannan.  
For educational and demonstration purposes.
