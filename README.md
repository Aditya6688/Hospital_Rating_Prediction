# 🏥 Hospital Rating Analysis & Prediction System

A comprehensive machine learning project that analyzes hospital performance data and predicts hospital ratings using multiple classification algorithms. This project provides insights into hospital quality metrics across the United States.

## 📊 Project Overview

This project analyzes hospital performance data from the Centers for Medicare & Medicaid Services (CMS) to understand factors influencing hospital ratings and predict whether a hospital will receive a high rating (4-5 stars) or low rating (1-3 stars).

### Key Features

- **📈 Comprehensive Data Analysis**: Analysis of 4,812 hospitals across the United States
- **🔍 Exploratory Data Analysis**: Detailed visualization of hospital characteristics and performance metrics
- **🤖 Machine Learning Models**: Implementation of multiple classification algorithms
- **📊 Interactive Visualizations**: Beautiful charts and graphs using Seaborn and Matplotlib
- **🎯 Predictive Modeling**: Binary classification to predict high vs low hospital ratings

## 🗂️ Dataset Information

The project uses the **Hospital General Information** dataset containing:

- **4,812 hospitals** across the United States
- **28 original features** including:
  - Hospital identification and location
  - Hospital type and ownership
  - Emergency services availability
  - Performance metrics and ratings
  - National comparison data

### Key Performance Metrics Analyzed

- Hospital Overall Rating (1-5 stars)
- Mortality National Comparison
- Safety of Care National Comparison
- Readmission National Comparison
- Patient Experience National Comparison
- Effectiveness of Care National Comparison
- Timeliness of Care National Comparison
- Efficient Use of Medical Imaging National Comparison

## 🚀 Installation & Setup

### Prerequisites

- Python 3.7 or higher
- Jupyter Notebook or JupyterLab
- Required Python packages (see requirements below)

### Installation Steps

1. **Clone or download the project files**
   ```bash
   # If using git
   git clone <repository-url>
   cd BTP_Hospital_Rating
   ```

2. **Install required packages**
   ```bash
   pip install numpy pandas matplotlib seaborn scikit-learn
   ```

3. **Launch Jupyter Notebook**
   ```bash
   jupyter notebook
   ```

4. **Open the notebook**
   - Open `Hospital_rating.ipynb` in Jupyter

## 📋 Required Dependencies

```python
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.4.0
seaborn>=0.11.0
scikit-learn>=1.0.0
jupyter>=1.0.0
```

## 🔬 Methodology

### Data Preprocessing
1. **Data Cleaning**: Removed missing values and irrelevant columns
2. **Feature Engineering**: 
   - One-hot encoding for categorical variables
   - State-wise dummy variables
   - Binary classification target (High: 4-5 stars, Low: 1-3 stars)
3. **Data Reduction**: Final dataset contains 2,297 hospitals with 74 features

### Machine Learning Models

The project implements and compares three classification algorithms:

1. **K-Nearest Neighbors (KNN)**
   - Optimized with k=19 neighbors
   - Accuracy: 86%
   - F1-Score: 0.85

2. **Support Vector Machine (SVM)**
   - Best performing model
   - Accuracy: 87%
   - F1-Score: 0.87

3. **Random Forest**
   - Ensemble method with 300 estimators
   - Accuracy: 86%
   - F1-Score: 0.85

## 📊 Results & Insights

### Model Performance Comparison

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| **Support Vector Machine** | **87%** | **87%** | **87%** | **87%** |
| Random Forest | 86% | 85% | 86% | 85% |
| K-Nearest Neighbors | 86% | 85% | 86% | 85% |

### Key Findings

- **Hospital Distribution**: Most hospitals are located in Texas, California, and Florida
- **Rating Distribution**: Majority of hospitals receive 3-star ratings
- **Emergency Services**: Most hospitals provide emergency services
- **Ownership**: Private non-profit hospitals are most common
- **Performance**: SVM model provides the best predictive performance

## 📈 Visualizations

The project includes comprehensive visualizations:

- Hospital type distribution across states
- Emergency services availability
- Hospital ownership patterns
- Rating distributions
- Performance metric comparisons
- State-wise hospital density
- Model performance analysis

## 🎯 Usage

1. **Run the complete analysis**:
   - Execute all cells in `Hospital_rating.ipynb`
   - View generated visualizations and insights

2. **Custom analysis**:
   - Modify parameters in the notebook
   - Add new visualizations
   - Test different machine learning models

3. **Predict hospital ratings**:
   - Use the trained models to predict ratings for new hospital data
   - Input hospital characteristics to get rating predictions

## 📁 Project Structure

```
BTP_Hospital_Rating/
├── Hospital_rating.ipynb          # Main analysis notebook
├── Hospital General Information.csv # Dataset
├── README.md                      # This file
```

## 🔧 Customization

### Adding New Models
```python
from sklearn.ensemble import GradientBoostingClassifier

# Add your model here
gb_model = GradientBoostingClassifier()
gb_model.fit(X_train, y_train)
gb_predictions = gb_model.predict(X_test)
```

### Modifying Visualizations
```python
# Customize plot styles
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
```

## 👨‍💻 Author

**BTP Hospital Rating Project**
- Created for academic/research purposes
- Focus on healthcare analytics and machine learning

## 🙏 Acknowledgments

- Centers for Medicare & Medicaid Services (CMS) for the dataset
- Scikit-learn community for machine learning tools
- Python data science ecosystem contributors

---

**Note**: This project is for educational and research purposes. Always verify results and consult healthcare professionals for medical decisions.

<div align="center">
  <p>Made with ❤️ for healthcare analytics</p>
</div>

