# Decision Tree Employment Prediction

A machine learning project that uses a **Decision Tree classifier** to predict whether candidates are suitable for employment.
This project is built using the **IBM HR Employee Attrition dataset** and documented in a Jupyter Notebook for clarity.

---

## 📌 Objectives

* Learn how to apply a Decision Tree algorithm to a real-world dataset.
* Understand preprocessing, training, and evaluation of a classification model.
* Visualize results and interpret feature importance.
* Provide both a **well-documented notebook (assignment submission)** and **modular Python code (professional repo)**.

---

## 📂 Repository Structure

```sh

DecisionTreeEmployment/
│
├── data/
│   └── WA_Fn-UseC_-HR-Employee-Attrition.csv   # Raw dataset
│
├── notebooks/
│   └── decision_tree_employment.ipynb          # Main Jupyter notebook (assignment documentation)
│
├── src/
│   ├── __init__.py
│   ├── preprocessing.py                        # Data preprocessing functions
│   ├── model.py                                # Model training and prediction functions
│   ├── evaluation.py                           # Evaluation and visualization functions
│   └── utils.py                                # Helper functions
│
├── outputs/
│   ├── figures/                                # Saved plots (decision tree, feature importance, etc.)
│   └── reports/                                # Exported PDF or HTML reports
│
├── tests/
│   ├── test_preprocessing.py                   # Tests for preprocessing
│   ├── test_model.py                           # Tests for model training
│   └── test_evaluation.py                      # Tests for evaluation metrics
│
├── requirements.txt                            # Dependencies
├── README.md                                   # Documentation (this file)
├── .gitignore                                  # Ignored files (venv, __pycache__, etc.)
└── LICENSE                                     # License file (optional)
```

---

## ⚙️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/DecisionTreeEmployment.git
cd DecisionTreeEmployment
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
# Activate it
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🚀 Usage

### Run the Notebook

1. Open `notebooks/decision_tree_employment.ipynb` in **VS Code** or **Jupyter Notebook**.
2. Run all cells to reproduce the workflow:

   * Data loading
   * Preprocessing
   * Exploratory Data Analysis (EDA)
   * Model training
   * Model evaluation
   * Predictions for new candidates
   * Feature importance visualization

### Run Python Scripts

If you want to use the modular code:

```python
from src.preprocessing import load_and_preprocess
from src.model import train_decision_tree, predict_candidates
from src.evaluation import evaluate_model

# Load dataset
X_train, X_test, y_train, y_test, le_target, feature_cols = load_and_preprocess("data/WA_Fn-UseC_-HR-Employee-Attrition.csv")

# Train model
model = train_decision_tree(X_train, y_train)

# Evaluate
evaluate_model(model, X_test, y_test, le_target)
```

---

## 📊 Results

* The Decision Tree achieved good accuracy on the test set.
* Confusion Matrix and Classification Report show performance breakdown.
* Predictions for new candidates match expectations:

  * Highly experienced & educated → Suitable.
  * Low experience & weak scores → Not suitable.
* Feature Importance shows **Interview Score** and **Years of Experience** are strong predictors.

---

## 🧪 Testing

Unit tests are included for preprocessing, modeling, and evaluation.
Run them using:

```bash
pytest tests/
```

---

## 📜 License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## 📌 Notes

* This project was developed as a **Data Analytics Assignment**.
* The **Notebook** is for submission and learning.
* The **src/** code is for clean modular development.
* You can extend this project by trying:

  * Random Forest (ensemble of trees).
  * Hyperparameter tuning for better accuracy.
  * Comparing with other classifiers (Logistic Regression, SVM).
