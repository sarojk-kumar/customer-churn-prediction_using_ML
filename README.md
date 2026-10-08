# customer-churn-prediction_using_ML
Customer Churn Analysis and Prediction using Python, Machine Learning, and Streamlit.
# 📊 Customer Churn Prediction Using Machine Learning

![Python](https://img.shields.io/badge/Python-3.x-blue)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Scikit--learn-orange)
![Streamlit](https://img.shields.io/badge/App-Streamlit-red)
![Status](https://img.shields.io/badge/Project-Completed-brightgreen)

## 📌 Project Overview

Customer churn means when a customer stops using a company's product or service.

The goal of this project is to predict whether a customer is likely to **churn (Yes)** or **not churn (No)** using Machine Learning.

In this project, customer data was cleaned, analyzed, prepared, and used to train multiple classification models. The models were compared and the best model was selected.

A simple **Streamlit web application** was also created where users can upload a customer CSV file and get churn predictions.

---

## 🎯 Project Objectives

- Clean the customer dataset.
- Perform basic Exploratory Data Analysis (EDA).
- Handle missing values.
- Convert categorical data into numerical data.
- Split data into training and testing sets.
- Train multiple Machine Learning models.
- Compare model performance.
- Select the best-performing model.
- Save the trained model.
- Build a Streamlit prediction application.
- Upload a new CSV file.
- Predict customer churn.
- Download the prediction results.

---

## 🔄 Project Workflow

```text
Customer Dataset
       ↓
Data Cleaning
       ↓
Exploratory Data Analysis
       ↓
Data Preprocessing
       ↓
Train-Test Split
       ↓
Train Multiple ML Models
       ↓
Model Evaluation
       ↓
Model Comparison
       ↓
Select Best Model
       ↓
Save Model
       ↓
Streamlit Application
       ↓
Upload New CSV
       ↓
Predict Churn
       ↓
Add Churn_Prediction
       ↓
Download CSV
```

---

# 📂 Dataset

The project uses a customer churn dataset containing information about customers, their services, charges, contracts, and churn status.

### Original Dataset

- **Rows:** 7043
- **Columns:** 21

### After Data Cleaning

- **Rows:** 7032
- **Columns:** 21
- **Missing records removed:** 11
- **Duplicate rows:** 0

### Main Features

- Customer ID
- Gender
- Senior Citizen
- Partner
- Dependents
- Tenure
- Phone Service
- Multiple Lines
- Internet Service
- Online Security
- Online Backup
- Device Protection
- Tech Support
- Streaming TV
- Streaming Movies
- Contract
- Paperless Billing
- Payment Method
- Monthly Charges
- Total Charges
- Churn

---

# 🧹 Data Cleaning

## TotalCharges

The `TotalCharges` column was converted from text to numeric format.

```python
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)
```

After conversion, 11 missing values were found.

These records were removed:

```python
df = df.dropna(subset=["TotalCharges"])
```

---

## Duplicate Check

Duplicate rows were checked.

```text
Duplicate rows = 0
```

---

## Customer ID

The customer ID was checked and all customer IDs were unique.

Because `customerID` is only an identifier and does not provide useful information for prediction, it was removed before model training.

```python
df = df.drop("customerID", axis=1)
```

---

# 📊 Exploratory Data Analysis

The churn distribution was checked.

| Churn | Customers |
|---|---:|
| No | 5163 |
| Yes | 1869 |

This shows that the dataset is somewhat imbalanced because there are more non-churn customers than churn customers.

### Gender Distribution

| Gender | Customers |
|---|---:|
| Male | 3549 |
| Female | 3483 |

The gender distribution is almost balanced.

---

# ⚙️ Data Preprocessing

## Feature and Target

The target variable is `Churn`.

```python
X = df.drop("Churn", axis=1)
y = df["Churn"]
```

The data contained:

```text
X = 7032 rows × 19 features
y = 7032 rows
```

---

## Target Encoding

The target values were converted into numerical values.

```python
y = y.map({
    "No": 0,
    "Yes": 1
})
```

Therefore:

```text
0 = No Churn
1 = Churn
```

---

## One-Hot Encoding

Categorical columns were converted into numerical columns using One-Hot Encoding.

```python
categorical_cols = X.select_dtypes(include="str").columns

X = pd.get_dummies(
    X,
    columns=categorical_cols,
    drop_first=True,
    dtype=int
)
```

After encoding:

```text
Rows    = 7032
Features = 30
```

---

# ✂️ Train-Test Split

The dataset was divided into training and testing data.

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
```

### Dataset Split

| Dataset | Shape |
|---|---|
| Training Data | 5625 × 30 |
| Testing Data | 1407 × 30 |

The training data was used to train the models, while the testing data was used to evaluate them on unseen data.

---

# 🤖 Machine Learning Models

Five approaches were tested:

1. Logistic Regression
2. Scaled Logistic Regression
3. Balanced Logistic Regression
4. Decision Tree
5. Random Forest

---

## 1️⃣ Logistic Regression

```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
```

### Result

**Accuracy: 80.45%**

Churn class performance:

- Precision: 0.65
- Recall: 0.58
- F1-score: 0.61

---

## 2️⃣ Scaled Logistic Regression

StandardScaler was used before training Logistic Regression.

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

scaled_model = LogisticRegression(max_iter=1000)

scaled_model.fit(X_train_scaled, y_train)

y_pred_scaled = scaled_model.predict(X_test_scaled)
```

### Result

**Accuracy: 80.38%**

The performance was almost the same as the original Logistic Regression.

---

## 3️⃣ Balanced Logistic Regression

Because the dataset contains more non-churn customers, a balanced Logistic Regression model was also tested.

```python
balanced_model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced"
)

balanced_model.fit(X_train_scaled, y_train)

y_pred_balanced = balanced_model.predict(X_test_scaled)
```

### Result

**Accuracy: 72.64%**

Churn recall increased to **80%**, meaning the model was better at identifying actual churn customers.

However, overall accuracy decreased.

---

## 4️⃣ Decision Tree

```python
from sklearn.tree import DecisionTreeClassifier

dt_model = DecisionTreeClassifier(
    random_state=42,
    max_depth=5
)

dt_model.fit(X_train, y_train)

y_pred_dt = dt_model.predict(X_test)
```

### Result

**Accuracy: 77.83%**

Churn F1-score: **0.59**

---

## 5️⃣ Random Forest

```python
from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_test)
```

### Result

**Accuracy: 78.96%**

Churn F1-score: **0.57**

---

# 📈 Model Comparison

| Model | Accuracy | Churn Recall | Churn F1 |
|---|---:|---:|---:|
| ⭐ Logistic Regression | **80.45%** | 58% | **0.61** |
| Scaled Logistic Regression | 80.38% | — | — |
| Balanced Logistic Regression | 72.64% | **80%** | **0.61** |
| Decision Tree | 77.83% | 60% | 0.59 |
| Random Forest | 78.96% | 52% | 0.57 |

---

# 🏆 Final Model

## Logistic Regression

Logistic Regression was selected as the final model.

### Why?

- Highest overall accuracy: **80.45%**
- Good churn F1-score: **0.61**
- Simple and easy to understand.
- Easy to explain during an interview.
- Easy to deploy.
- Good performance compared with the other tested models.

---

# 💾 Model Saving

The final Logistic Regression model was saved using Joblib.

```python
import joblib

joblib.dump(
    model,
    "customer_churn_model.pkl"
)
```

Saved model:

```text
customer_churn_model.pkl
```

The saved model can be loaded later without training it again.

---

# 🌐 Streamlit Application

A Streamlit web application was created to use the trained model.

### Application Features

- Upload customer CSV.
- Display uploaded data.
- Check required columns.
- Prepare customer data.
- Perform one-hot encoding.
- Match model features.
- Predict customer churn.
- Add `Churn_Prediction`.
- Download prediction CSV.

---

# 🖥️ Application Workflow

```text
Upload CSV
     ↓
Check Required Columns
     ↓
Data Preprocessing
     ↓
One-Hot Encoding
     ↓
Match Training Features
     ↓
ML Model Prediction
     ↓
Churn_Prediction
     ↓
Download CSV
```

---

# 📤 Input

The uploaded CSV should contain the same customer input columns used during training.

The application expects columns such as:

```text
customerID
gender
SeniorCitizen
Partner
Dependents
tenure
PhoneService
MultipleLines
InternetService
OnlineSecurity
OnlineBackup
DeviceProtection
TechSupport
StreamingTV
StreamingMovies
Contract
PaperlessBilling
PaymentMethod
MonthlyCharges
TotalCharges
```

---

# 📥 Output

The application adds a new column:

```text
Churn_Prediction
```

Possible values:

```text
Yes
No
```

Example:

| customerID | gender | tenure | MonthlyCharges | Churn_Prediction |
|---|---|---:|---:|---|
| C001 | Male | 12 | 70.50 | Yes |
| C002 | Female | 48 | 55.20 | No |
| C003 | Male | 5 | 90.10 | Yes |

The final file can be downloaded as:

```text
customer_churn_predictions.csv
```

---

# 🛠️ Technologies Used

- **Python**
- **Pandas**
- **NumPy**
- **Scikit-learn**
- **Joblib**
- **Streamlit**
- **Jupyter Notebook**
- **Git & GitHub**

---

# 📁 Project Structure

```text
Customer-Churn-Project/
│
├── app.py
├── customer_churn_model.pkl
├── README.md
├── requirements.txt
└── customerchurn.csv
```

> The dataset file can also be kept outside the GitHub repository if required.

---

# ⚙️ Installation

Clone the repository:

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

Go to the project folder:

```bash
cd Customer-Churn-Project
```

Install the required libraries:

```bash
pip install -r requirements.txt
```

---

# ▶️ Run the Application

Run the Streamlit application:

```bash
streamlit run app.py
```

The application will open in your web browser.

Upload the customer CSV file and generate predictions.

---

# 📦 requirements.txt

```text
pandas
numpy
scikit-learn
joblib
streamlit
```

---

# 📸 Screenshots

Add your project screenshots here after taking them from the Streamlit application.

### Streamlit Home Page

```text
![Streamlit Home Page](screenshots/home.png)
```

### Uploaded Customer Data

```text
![Uploaded Data](screenshots/uploaded-data.png)
```

### Prediction Result

```text
![Prediction Result](screenshots/prediction.png)
```

### Download Prediction

```text
![Download Result](screenshots/download.png)
```

---

# 📌 Key Results

| Metric | Result |
|---|---:|
| Original Dataset | 7043 rows |
| Cleaned Dataset | 7032 rows |
| Missing Records Removed | 11 |
| Duplicate Rows | 0 |
| Final Features | 30 |
| Training Records | 5625 |
| Testing Records | 1407 |
| Models Tested | 5 |
| Best Model | Logistic Regression |
| Best Accuracy | **80.45%** |
| Churn F1-score | **0.61** |

---

# 🎓 What I Learned

Through this project, I learned how to:

- Work with a real-world customer dataset.
- Clean missing data.
- Check duplicate records.
- Perform basic EDA.
- Prepare features and target variables.
- Encode categorical variables.
- Split data into training and testing sets.
- Train classification models.
- Compare different ML models.
- Understand accuracy, precision, recall, and F1-score.
- Save a trained ML model.
- Load a saved model.
- Build a Streamlit application.
- Perform predictions on a new CSV file.
- Generate a downloadable prediction file.

---

# 🚀 Future Improvements

In the future, this project can be improved by:

- Hyperparameter tuning.
- Testing additional ML algorithms.
- Improving churn recall.
- Adding churn probability.
- Adding interactive charts.
- Adding a model performance dashboard.
- Handling more types of input data.
- Deploying the Streamlit application online.

---

# 👩‍💻 Project Author

**Saroj Kumar**

B.Tech Computer Science Engineering

Skills demonstrated in this project:

`Python` • `Pandas` • `Machine Learning` • `Scikit-learn` • `Data Analysis` • `Streamlit` • `GitHub`

---

# ⭐ Conclusion

This project demonstrates a complete beginner-friendly Machine Learning workflow for customer churn prediction.

The data was cleaned and analyzed, categorical variables were encoded, and multiple classification models were trained and compared.

**Logistic Regression achieved the highest accuracy of 80.45% and was selected as the final model.**

The trained model was saved and integrated into a Streamlit application. The application can accept a new customer CSV file, predict customer churn, add the prediction to the data, and provide the final CSV file for download.
