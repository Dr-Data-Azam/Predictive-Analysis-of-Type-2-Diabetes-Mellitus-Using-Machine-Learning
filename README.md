# Predictive-Analysis-of-Type-2-Diabetes-Mellitus-Using-Machine-Learning
This project leverages machine learning to predict the risk of developing Type 2 Diabetes Mellitus (T2DM) using patient demographics, clinical parameters, and medical history. The goal is to provide healthcare providers with a reliable tool for identifying high-risk individuals and implementing targeted interventions for diabetes prevention.

## Project Overview:-
Type 2 Diabetes Mellitus is a common metabolic disorder with significant health implications, including cardiovascular disease, kidney damage, and neuropathy. This project explores the use of machine learning models to predict T2DM risk, focusing on enhancing early detection and personalized care strategies.

### Key Features:-
Data Source: DiaHealth Dataset, consisting of 5,386 cleaned patient records.
Features Used: Age, gender, glucose levels, BMI, blood pressure, and medical history (e.g., family diabetes, stroke, cardiovascular disease).
### Tools and Technologies:
Programming: Python
Libraries: Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn
Database: PostgreSQL
Modeling: Logistic Regression, Random Forest, Support Vector Machine (SVM)
## Methodology
### Data Cleaning and Preprocessing:
Removal of outliers using IQR.
Handling class imbalance with SMOTE.
Removal of multicollinearity based on VIF values.
### Model Training:
Logistic Regression, Random Forest, and SVM.
Hyperparameter tuning with GridSearchCV.
Custom thresholds for improved minority class recall.
### Evaluation:
Metrics: Precision, Recall, F1-Score, ROC-AUC.
Best-performing model: Random Forest (AUC = 0.89).
## Key Findings
Significant predictors: Glucose, BMI, systolic BP, age.
Insignificant predictors: Family diabetes and hypertension.
Random Forest outperformed other models, showing robustness for imbalanced datasets.
## Visualizations
Comparative ROC curves for all models.
Confusion matrices to analyze prediction performance.
Feature importance from the Random Forest model.
## Challenges and Limitations
Data imbalance (6.4% diabetic).
Lack of lifestyle-related variables.
Dataset limitation to Bangladeshi population.
## Future Work
Incorporate additional predictors like lifestyle habits and genetic predisposition.
Explore advanced models (e.g., XGBoost, deep learning).
Conduct cross-validation and global population studies for broader applicability.
## Contributors
Sheikh Azam Uddin: Data cleaning, exploratory data analysis, model training.
Sai Kumar Vemula: Model training, refining, comparison.
Michael Perry: SQL database management, data import, report creation.
#### For more details, refer to the full project documentation in this repository.
