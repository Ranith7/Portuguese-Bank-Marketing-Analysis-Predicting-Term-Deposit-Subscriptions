# 📊 Portuguese Bank Marketing Prediction Project

This project focuses on predicting whether a customer will subscribe to a term deposit based on the Portuguese Bank Marketing dataset. The dataset comes from a direct marketing campaign, and the goal is to assist the bank in identifying potential subscribers using machine learning models.

---

## 🗂️ Project Structure

Portuguese-Bank-Marketing-Analysis/
│
├── Portuguese Prediction/                     # Main folder for model training and evaluation
│   ├── Dataset/
│   │   └── bank-additional-full.csv           # The dataset used for prediction
│   ├── Notebook/
│   │   └── Portuguese Prediction.ipynb        # Jupyter Notebook with complete analysis
│   └── Introduction/
│       └── Project_Intro.docx                 # Project introduction and business understanding

---

## 📌 Problem Statement

The Portuguese bank conducted a telemarketing campaign to promote term deposits. The goal is to develop a predictive model to identify customers who are likely to subscribe, thus improving the efficiency and cost-effectiveness of future campaigns.

---

## 📂 Dataset Overview

- **Source**: UCI Machine Learning Repository
- **File**: `bank-additional-full.csv`
- **Records**: ~41,000
- **Features**: Demographics, past campaign interactions, call duration, communication type, etc.
- **Target Variable**: `y` — whether the client subscribed (`yes` or `no`)

---

## 🛠️ Technologies Used

- Python
- Jupyter Notebook
- Pandas, NumPy
- Scikit-learn
- Matplotlib, Seaborn
- XGBoost, Gradient Boosting

---

## 📈 Project Workflow

1. **Data Preprocessing**
   - Handled missing values
   - Label encoding and one-hot encoding
   - Feature scaling

2. **Exploratory Data Analysis (EDA)**
   - Visualized trends and patterns using charts
   - Analyzed correlation between features and subscription status

3. **Model Building**
   - Trained multiple classification models:
     - Logistic Regression
     - Support Vector Machine (SVM)
     - K-Nearest Neighbors (KNN)
     - Decision Tree
     - Gradient Boosting
     - XGBoost
   - Used both imbalanced and balanced datasets (SMOTE)

4. **Model Evaluation**
   - Compared performance using metrics like Accuracy, Precision, Recall, F1-Score
   - Gradient Boosting (balanced) achieved the best F1-score for Class 1 (subscribers)

5. **Business Recommendations**
   - Target campaigns during the months of March, September, October, and December
   - Focus on job, education, and call duration features
   - Use mobile phones for better engagement
   - Optimize calling strategies based on model predictions

---

## ✅ Best Performing Model

| Model               | Precision (Class 1) | Recall (Class 1) | F1 Score (Class 1) | Accuracy |
|--------------------|---------------------|------------------|--------------------|----------|
| Gradient Boosting (balanced) | 34%               | 52%             | **41%**           | 83%      |

---

## 📄 Files Description

| Folder/File                     | Description |
|--------------------------------|-------------|
| `Dataset/bank-additional-full.csv` | Raw dataset used for the project |
| `Notebook/Portuguese Prediction.ipynb` | Complete notebook with EDA, model building, evaluation |
| `Introduction/Project_Intro.docx` | Introduction and business understanding document |

---

## 🎯 Key Takeaways

- The dataset is highly imbalanced; hence class balancing is necessary.
- Gradient Boosting with balanced data performs the best in identifying potential subscribers.
- Data-driven marketing decisions can significantly improve success rates.

> ✨ If you found this helpful, leave a ⭐ on the repository!


