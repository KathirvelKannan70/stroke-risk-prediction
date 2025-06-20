
## Heart Disease Prediction using Logistic Regression**

This project demonstrates how to build a machine learning model to predict the presence of heart disease using a logistic regression algorithm. The entire workflow — from data loading to prediction — is implemented within a single Jupyter Notebook.

---
### 1. Requirements

Make sure Python is installed, then install the necessary libraries:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

### 2. Run the Notebook

Open the notebook with Jupyter:

```bash
jupyter notebook heart_disease_prediction.ipynb
```

Follow the notebook steps to:

- Load and explore the dataset
- Preprocess the data (handling missing values, encoding)
- Train a **Logistic Regression** model
- Evaluate performance with accuracy
- Test the model with custom input values

---

## Sample Prediction

The notebook includes an example like:

```
input_data = (51,1,3,125,213,0,0,125,1,1.4,2,1,2)

# Converting it into a numpy array
input_data = np.asarray(input_data) 

# Reshaping into requred input form
input_data = input_data.reshape(1,-1) 

# With the input I am predicting 
prediction = model.predict(input_data)
prediction 

if prediction[0] == 0:
    print("Good News the patient does'nt have any heart disease")
else: 
    print("The Patient should visit the doctor")
```

---

## Dataset Features

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

## Model Summary

- **Model Used**: Logistic Regression
- **Evaluation Metrics**: Accuracy, Confusion Matrix, Precision, Recall
- **Approach**: All steps are done within one notebook, no external files used

---

## Author

Developed by Kathirvel Kannan.  
For educational and demonstration purposes.
