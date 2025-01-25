"""
A library of functions to find customers who are likely to churn.

This library contains utility functions for loading, preprocessing, and analyzing
customer data to predict churn. It includes functions for:
- Data importing and EDA
- Feature engineering and encoding
- Model training (Random Forest and Logistic Regression)
- Model evaluation and result visualization

Author: Fabian Bruckschen
Date: 17.01.25
"""

# import libraries
import os
import logging
import warnings
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import shap
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import RocCurveDisplay, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV

# logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filename="logs/churn.log",
)

# setup
logger = logging.getLogger(__name__)
os.environ["QT_QPA_PLATFORM"] = "offscreen"
warnings.filterwarnings("ignore", category=ConvergenceWarning)


def import_data(
    pth: str = "./data/bank_data.csv", recreate_target: bool = True
) -> pd.DataFrame:
    """
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            df: pandas dataframe
    """
    df = pd.read_csv(pth)
    if recreate_target:
        try:
            df["Churn"] = df["Attrition_Flag"].apply(
                lambda val: 0 if val == "Existing Customer" else 1
            )
            df = df.drop(["Attrition_Flag"], axis=1)
        except KeyError:
            logger.info("Target variable could not be created!")
    return df


def perform_eda(df: pd.DataFrame, output_dir: str = "images/eda") -> None:
    """
    perform eda on df and save figures to images folder
    input:
            df: pandas dataframe

    output:
            None
    """
    # plot settings
    os.makedirs(output_dir, exist_ok=True)

    # Create the plots and save them as variables
    fig0, ax0 = plt.subplots(figsize=(20, 10))
    df["Churn"].hist(ax=ax0)
    ax0.set_title("Churn Distribution")

    fig1, ax1 = plt.subplots(figsize=(20, 10))
    df["Customer_Age"].hist(ax=ax1)
    ax1.set_title("Customer Age Distribution")

    fig2, ax2 = plt.subplots(figsize=(20, 10))
    df["Marital_Status"].value_counts("normalize").plot(kind="bar", ax=ax2)
    ax2.set_title("Marital Status Distribution")

    fig3, ax3 = plt.subplots(figsize=(20, 10))
    sns.histplot(df["Total_Trans_Ct"], stat="density", kde=True, ax=ax3)
    ax3.set_title("Total Transaction Count Distribution")

    fig4, ax4 = plt.subplots(figsize=(20, 10))
    correlation = df.select_dtypes(include="number").corr()
    sns.heatmap(correlation, annot=False, cmap="Dark2_r", linewidths=2, ax=ax4)
    ax4.set_title("Feature Correlation Heatmap")

    # Create a list of tuples containing the figures and their filenames
    plots = [
        (fig0, "churn_histogram.png"),
        (fig1, "customer_age_histogram.png"),
        (fig2, "marital_status_bar.png"),
        (fig3, "total_trans_ct_density.png"),
        (fig4, "correlation_heatmap.png"),
    ]

    # Save all plots
    for fig, filename in plots:
        fig.savefig(os.path.join(output_dir, filename))
        plt.close(fig)


def encoder_helper(
    df: pd.DataFrame, category_lst: list[str], response: str = "Churn"
) -> pd.DataFrame:
    """
    helper function to turn each categorical column into a new column with
    proportion of churn for each category - associated with cell 15 from the notebook

    input:
            df: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name [optional argument that could be used for
                naming variables or index y column]

    output:
            df: pandas dataframe with new columns for
    """
    for cat in category_lst:
        groups = df.groupby(cat)[response].mean()
        df[f"{cat}_{response}"] = [groups.loc[val] for val in df[cat]]
    return df


def perform_feature_engineering(
    df: pd.DataFrame,
    columns: Optional[list[str]],
    response: str = "Churn",
    test_size: float = 0.3,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    input:
              df: pandas dataframe
              columns: list of column names to select for features (if None, uses all columns)
              response: string of response name [optional argument that could be used for
                naming variables or index y column]
              test_size: proportion of dataset to include in the test split
              random_state: controls the shuffling applied to the data before applying the split
              response: string of response name [optional argument that could be used for
                naming variables or index y column]

    output:
              x_data_train: X training data
              x_data_test: X testing data
              y_train: y training data
              y_test: y testing data
    """
    y = df[response]
    x_data = pd.DataFrame()
    if columns:
        x_data[columns] = df[columns]
    else:
        x_data = df
    return train_test_split(x_data, y, test_size=test_size, random_state=random_state)


def classification_report_image(
    y_train: pd.Series,
    y_test: pd.Series,
    y_train_preds: pd.Series,
    y_test_preds: pd.Series,
    model_name: str,
    output_pth: str = "images/results/",
) -> None:
    """
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions
            y_test_preds_rf: test predictions
            model_name: name of model
            output_pth: path to store the resulting images

    output:
             None
    """
    plt.figure(figsize=(10, 10))

    # Plot train results
    plt.text(
        0.01,
        1,
        f"{model_name} Train",
        {"fontsize": 10},
        fontproperties="monospace",
    )
    plt.text(
        0.01,
        0.8,
        str(classification_report(y_train, y_train_preds)),
        {"fontsize": 10},
        fontproperties="monospace",
    )

    # Plot test results
    plt.text(
        0.01,
        0.5,
        f"{model_name} Test",
        {"fontsize": 10},
        fontproperties="monospace",
    )
    plt.text(
        0.01,
        0.3,
        str(classification_report(y_test, y_test_preds)),
        {"fontsize": 10},
        fontproperties="monospace",
    )

    plt.axis("off")

    # Save the plot
    plt.savefig(f"{output_pth}/{model_name.lower().replace(' ', '_')}_results.png")
    plt.close()


def feature_importance_plot(
    model: RandomForestClassifier,
    x_data: pd.DataFrame,
    output_pth: str = "images/results/",
):
    """
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    """

    importances = model.best_estimator_.feature_importances_
    indices = np.argsort(importances)[::-1]
    names = [x_data.columns[i] for i in indices]

    os.makedirs(output_pth, exist_ok=True)
    plt.figure(figsize=(20, 5))
    plt.title("Feature Importance")
    plt.ylabel("Importance")
    plt.bar(range(x_data.shape[1]), importances[indices])
    plt.xticks(range(x_data.shape[1]), names, rotation=90)

    plt.tight_layout()
    plt.savefig(os.path.join(output_pth, "feature_importances.png"))
    plt.close()


def train_models(
    x_data_train: pd.DataFrame,
    x_data_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    output_plots_pth: str = "images/results",
    output_models_pth: str = "models",
) -> tuple[
    RandomForestClassifier,
    LogisticRegression,
    pd.Series,
    pd.Series,
    pd.Series,
    pd.Series,
    pd.Series,
]:
    """
    train, store model results: images + scores, and store models
    input:
              x_data_train: X training data
              x_data_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    """
    # LR
    # Try different solvers in order of preference due to potential convergence errors
    solvers = ["lbfgs", "liblinear", "newton-cg", "sag", "saga"]
    for solver in solvers:
        try:
            lrc = LogisticRegression(solver=solver, max_iter=3000)
            lrc.fit(x_data_train, y_train)
            y_train_preds_lr = lrc.predict(x_data_train)
            y_test_preds_lr = lrc.predict(x_data_test)
            break  # Exit the loop if fitting succeeds
        except ConvergenceWarning:
            logger.info("Solver %s failed to converge. Trying next solver.", solver)
        except ValueError as e:
            if "Solver" in str(e):
                logger.info(
                    "Solver %s encountered an error: %s. Trying next solver.", solver, e
                )
            else:
                raise  # Re-raise if it's not a solver-related ValueError
        except Exception as e:
            logger.error("Unexpected error with solver %s: %s", solver, e)
            raise  # Re-raise unexpected exceptions

    # RF
    param_grid = {
        "n_estimators": [200, 500],
        "max_features": ["auto", "sqrt"],
        "max_depth": [4, 5, 100],
        "criterion": ["gini", "entropy"],
    }
    rfc = RandomForestClassifier(random_state=42)
    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    cv_rfc.fit(x_data_train, y_train)
    y_train_preds_rf = cv_rfc.best_estimator_.predict(x_data_train)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(x_data_test)

    # models
    joblib.dump(cv_rfc.best_estimator_, f"{output_models_pth}/rfc_model.pkl")
    joblib.dump(lrc, f"{output_models_pth}/logistic_model.pkl")

    # plots
    plt.figure(figsize=(15, 8))
    ax = plt.gca()
    RocCurveDisplay.from_estimator(
        cv_rfc.best_estimator_, x_data_test, y_test, ax=ax, alpha=0.8
    )
    RocCurveDisplay.from_estimator(lrc, x_data_test, y_test, ax=ax, alpha=0.8)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC) Curve")
    plt.savefig(f"{output_plots_pth}/roc_curve.png")
    plt.close()

    plt.figure(figsize=(15, 8))
    explainer = shap.TreeExplainer(cv_rfc.best_estimator_)
    shap_values = explainer.shap_values(x_data_test)
    shap.summary_plot(shap_values, x_data_test, plot_type="bar", show=False)
    plt.savefig(f"{output_plots_pth}/shap_importance.png")
    plt.close()

    return (
        cv_rfc,
        lrc,
        y_train_preds_lr,
        y_test_preds_lr,
        y_train_preds_rf,
        y_test_preds_rf,
    )


if __name__ == "__main__":
    # Load and process data
    df = import_data()

    # Perform EDA
    perform_eda(df)

    # Encode categorical variables
    cat_columns = [
        "Gender",
        "Education_Level",
        "Marital_Status",
        "Income_Category",
        "Card_Category",
    ]
    df = encoder_helper(df, cat_columns)

    # Select features for modeling
    feature_cols = [
        "Customer_Age",
        "Dependent_count",
        "Months_on_book",
        "Total_Relationship_Count",
        "Months_Inactive_12_mon",
        "Contacts_Count_12_mon",
        "Credit_Limit",
        "Total_Revolving_Bal",
        "Avg_Open_To_Buy",
        "Total_Amt_Chng_Q4_Q1",
        "Total_Trans_Amt",
        "Total_Trans_Ct",
        "Total_Ct_Chng_Q4_Q1",
        "Avg_Utilization_Ratio",
        "Gender_Churn",
        "Education_Level_Churn",
        "Marital_Status_Churn",
        "Income_Category_Churn",
        "Card_Category_Churn",
    ]

    # Perform feature engineering and split data
    X_train, X_test, y_train, y_test = perform_feature_engineering(df, feature_cols)

    # Train and evaluate models
    rfc, lrc, y_train_preds_lr, y_test_preds_lr, y_train_preds_rf, y_test_preds_rf = (
        train_models(X_train, X_test, y_train, y_test)
    )

    # Result plots
    feature_importance_plot(rfc, pd.concat([X_train, X_test]))
    classification_report_image(
        y_train,
        y_test,
        y_train_preds_lr,
        y_test_preds_lr,
        model_name="Logistic Regression",
    )
    classification_report_image(
        y_train, y_test, y_train_preds_lr, y_test_preds_lr, model_name="Random Forest"
    )

    logger.info("Churn prediction model training completed successfully")
