# Data Analytics - Assignment 1  

## Practical Question: Machine Learning Using Decision Tree on Employment Dataset  

### Objective  

You are provided with an **Employment Dataset** containing information about candidates who applied for jobs.  
Your task is to build a **Decision Tree Classification Model** to predict whether a candidate should be employed or not based on various features.  

Dataset link: [IBM HR Analytics Attrition Dataset](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attritiondataset)  

- Form groups of **4–6 members** to complete the task.  

---

## Dataset Description  

Each row in the dataset represents a job applicant. The dataset includes the following features:  

- **age** – Age of the employee (in years)  
- **education_level** – Highest education level attained (e.g., High School, Bachelor’s, Master’s, PhD)  
- **years_of_experience** – Total professional experience (in years)  
- **technical_test_score** – Score from technical assessment (out of 100)  
- **interview_score** – Score from interview (out of 10)  
- **previous_employment** – Previous work experience (Yes/No)  
- **suitable_for_employment (Target)** – Indicates suitability for employment (Yes/No)  

---

## Tasks to Perform  

### 1. Data Loading and Exploration  

- Load the dataset using Python libraries (e.g., pandas).  
- Display the first few rows of the dataset.  
- Perform basic EDA (Exploratory Data Analysis):  
  - Check for null values  
  - Inspect data types  
  - Explore feature distributions  

### 2. Data Preprocessing  

- Convert categorical variables into numeric format (e.g., one-hot encoding or label encoding).  
- Split the dataset into **training (80%)** and **testing (20%)** sets.  

### 3. Model Building  

- Train a **Decision Tree Classifier** using the training data to predict `suitable_for_employment`.  

### 4. Model Visualization  

- Visualize the decision tree using tools such as `plot_tree()` or **Graphviz**.  

### 5. Model Testing and Prediction  

- Predict labels for the test dataset.  
- Test the model with **3 hypothetical candidate profiles** and interpret predictions.  

### 6. Model Evaluation  

- Evaluate the model using:  
  - Accuracy Score  
  - Confusion Matrix  
  - Classification Report (Precision, Recall, F1-Score)  

## Bonus Task (Optional)  

- Perform **feature importance analysis** to determine which features contribute most to the employment decision.  

## 📦 Required Libraries  

- pandas  
- numpy  
- scikit-learn  
- matplotlib  
- seaborn  

## Expected Output  

- Clear and well-commented Python code  
- Visualized decision tree  
- Model performance metrics  
- Interpretation of predictions  
