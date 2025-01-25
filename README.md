# Predict Customer Churn

- Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree Udacity

## Project Description
This project implements a machine learning solution to predict customer churn for a bank. It includes a complete ML pipeline from data loading and preprocessing to model training and evaluation, with comprehensive logging and testing functionality.

**Key features:**
- Data preprocessing and EDA with automated visualizations
- Feature engineering and encoding
- Model training (Random Forest and Logistic Regression)
- Model evaluation with classification reports and ROC curves
- Comprehensive testing suite
- Detailed logging system

## Files and Data Description

### Main Files:
- `churn_library.py`: Main library containing all ML pipeline functions
- `churn_script_logging_and_tests.py`: Testing and logging implementation
- `requirements.txt`: Project dependencies

### Key Functions in churn_library.py:
- `import_data()`: Imports and validates the bank customer data
- `perform_eda()`: Performs exploratory data analysis with automated plots
- `encoder_helper()`: Handles categorical variable encoding
- `perform_feature_engineering()`: Prepares features for modeling
- `train_models()`: Trains and saves Random Forest and Logistic Regression models
- `classification_report_image()`: Generates classification report visualizations
- `feature_importance_plot()`: Creates feature importance plot

### Data:
The project uses the bank_data.csv file which should contain the following key features:
- Customer demographic information
- Banking relationship data
- Transaction history
- Churn status (target variable)

## Running Files

### Environment Setup
1. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```
2. Install required packages:
```bash
pip install -r requirements.txt
```
### Running the ML Pipeline
```bash
python churn_library.py
```
This will:

- Load and process the data
- Perform EDA and generate visualization plots
- Train and save the models
- Generate performance metrics and plots

### Running Tests
```bash
python churn_script_logging_and_tests.py
```
This will:

- Test all functions in the ML pipeline
- Generate logs in ./logs/churn_library.log
- Create test outputs in the respective directories

### Expected Outputs

After running the tests, you should find:

EDA plots in `./images/eda/`:

- Churn distribution
- Customer age distribution
- Marital status distribution
- Transaction count distribution
- Correlation heatmap

Results in `./images/results/`:

- ROC curves
- Feature importance plots
- Classification reports

Trained models in `./models/`:

- rfc_model.pkl
- logistic_model.pkl
- Logs in ./logs/churn_library.log

### Logging
The logging system provides detailed information about:

- Function execution success/failure
- Data validation results
- Model training progress
- Test results
Logs can be found in `./logs/churn_library.log` after running either the main pipeline or tests.

### License
This project is licensed under the MIT License - see the LICENSE file for details.