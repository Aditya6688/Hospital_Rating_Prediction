# CMS Hospital Ratings Analysis

## Overview
This project analyzes and models hospital quality ratings using data from the Centers for Medicare & Medicaid Services (CMS). The analysis is performed in the Jupyter notebook `cms-hospital-ratings.ipynb` and covers data preparation, cleaning, feature engineering, modeling, evaluation, and interpretation of results.

## Data Sources
The notebook uses several CSV files from the CMS Hospital Compare datasets, including:
- Readmissions and Deaths - Hospital.csv
- Complications - Hospital.csv
- Healthcare-Associated Infections - Hospital.csv
- HCAHPS - Hospital.csv
- Outpatient Imaging Efficiency - Hospital.csv
- Timely and Effective Care - Hospital.csv

These files are expected to be located in the `../input/capstone-project-cms-hospital-ratings/Hospital_Revised_FlatFiles_20161110/` directory relative to the notebook.

## Main Steps
1. **Data Preparation & Understanding**: Load and inspect multiple CMS datasets, focusing on key quality measures (readmission, mortality, safety, patient experience, medical imaging, timeliness, effectiveness).
2. **Data Cleaning**: Handle missing values, convert data types, and standardize features. Outlier treatment is also performed.
3. **Feature Engineering**: Pivot and merge datasets to create wide-format tables for each quality domain. Calculate composite scores using statistical and factorial analysis.
4. **Modeling**: Train and evaluate several machine learning models (Random Forest, Gradient Boosting, Neural Networks, KNN, Logistic Regression, Decision Trees) to predict hospital ratings.
5. **Evaluation**: Assess model performance using metrics such as accuracy, precision, recall, F1-score, ROC-AUC, and confusion matrices.
6. **Interpretation**: Use feature importance, permutation importance, LIME, ELI5, and SHAP for model interpretability and to understand key drivers of hospital ratings.

## Dependencies
The notebook requires the following Python packages:
- pandas
- numpy
- seaborn
- matplotlib
- plotnine
- scikit-learn
- factor_analyzer
- rfpimp
- lime
- eli5
- shap

Install dependencies using pip:
```bash
pip install pandas numpy seaborn matplotlib plotnine scikit-learn factor_analyzer rfpimp lime eli5 shap
```

## Usage
1. Place the required CMS data files in the expected directory structure.
2. Open `cms-hospital-ratings.ipynb` in Jupyter Notebook or JupyterLab.
3. Run the notebook cells sequentially to reproduce the analysis, modeling, and interpretation steps.

## Results & Interpretation
The notebook provides:
- Cleaned and merged datasets for each hospital quality domain
- Composite scores for readmission, mortality, safety, experience, medical, timeliness, and effectiveness
- Machine learning models to predict hospital ratings
- Visualizations and interpretability analyses to understand feature importance and model decisions

## Author
*This README was generated based on the code and comments in the notebook. Please refer to the notebook for detailed code, outputs, and further explanations.*
