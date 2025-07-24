# 🧠 Diabetes Prediction using Machine Learning

This project predicts whether a person is diabetic based on medical diagnostic measurements. It uses machine learning classification algorithms such as **Logistic Regression**, **Support Vector Machine (SVM)**, **K-Nearest Neighbors (KNN)**, and **Multilayer Perceptron (MLP Neural Network)** trained on the **Pima Indians Diabetes Dataset**.

---

## 📁 Dataset

- **Source**: Pima Indians Diabetes Dataset
- **Features**:
  - Pregnancies
  - Glucose
  - BloodPressure
  - SkinThickness
  - Insulin
  - BMI
  - DiabetesPedigreeFunction
  - Age
  - Outcome (Target Variable: 0 = Non-Diabetic, 1 = Diabetic)

---

## 📊 Workflow

1. **Data Preprocessing**  
   - Handled missing/zero values by replacing them with mean or median.
   - Feature scaling using `StandardScaler`.

2. **Exploratory Data Analysis (EDA)**  
   - Visualized feature distributions using **Seaborn** (`kdeplot`, `violinplot`, `heatmap`).
   - Plotted correlation heatmaps to identify significant features.

3. **Model Training & Evaluation**  
   - Trained multiple classifiers:
     - Logistic Regression
     - Support Vector Machine (SVM)
     - K-Nearest Neighbors (KNN)
     - Decision Tree
     - Multilayer Perceptron (MLP)
   - Evaluated models using:
     - Accuracy Score
     - Confusion Matrix
     - Precision, Recall, F1-score

4. **Best Result**  
   - **MLP Classifier (with scaling)** achieved **72% test accuracy**.

---

## 🛠️ Tech Stack

- Python
- Pandas, NumPy
- Seaborn, Matplotlib
- Scikit-learn

---

