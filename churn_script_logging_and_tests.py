"""
Test script for churn_library.py functions.
Contains unit tests and logging functionality.

Author: Fabian Bruckschen
Date: 25.01.25
"""

import os
import logging

import pandas as pd
from churn_library import (
    import_data,
    perform_eda,
    encoder_helper,
    perform_feature_engineering,
    train_models,
    classification_report_image,
    feature_importance_plot,
)

# Configure logging
logging.basicConfig(
    filename="./logs/churn_library.log",
    level=logging.INFO,
    filemode="w",
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


def test_import():
    """
    Test data import functionality
    """
    try:
        data = import_data("./data/bank_data.csv")
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing import_data: The file wasn't found")
        raise err

    try:
        assert data.shape[0] > 0
        assert data.shape[1] > 0
        assert "Churn" in data.columns
        logging.info(
            "Data import testing: Data has %d rows and %d columns",
            data.shape[0],
            data.shape[1],
        )
    except AssertionError as err:
        logging.error(
            "Testing import_data: The file doesn't appear to have the correct format"
        )
        raise err

    return data


def test_eda(data):
    """
    Test perform eda functionality
    """
    try:
        perform_eda(data)
        logging.info("Testing perform_eda: SUCCESS")

        # Check if EDA output files exist
        eda_plots = [
            "churn_histogram.png",
            "customer_age_histogram.png",
            "marital_status_bar.png",
            "total_trans_ct_density.png",
            "correlation_heatmap.png",
        ]

        for plot in eda_plots:
            assert os.path.exists(os.path.join("images/eda", plot))
        logging.info("EDA testing: All plot files were successfully created")

    except AssertionError as err:
        logging.error("Testing perform_eda: Missing expected plot files")
        raise err
    except Exception as err:
        logging.error("Testing perform_eda: Function failed with error: %s", err)
        raise err


def test_encoder_helper(data):
    """
    Test encoder helper functionality
    """
    try:
        category_lst = [
            "Gender",
            "Education_Level",
            "Marital_Status",
            "Income_Category",
            "Card_Category",
        ]

        data_encoded = encoder_helper(data, category_lst)

        # Check if new encoded columns exist
        for category in category_lst:
            assert f"{category}_Churn" in data_encoded.columns

        # Check if encoded values are within expected range (0 to 1)
        for category in category_lst:
            assert 0 <= data_encoded[f"{category}_Churn"].max() <= 1
            assert 0 <= data_encoded[f"{category}_Churn"].min() <= 1

        logging.info("Testing encoder_helper: SUCCESS")
        return data_encoded

    except AssertionError as err:
        logging.error(
            "Testing encoder_helper: Encoded columns missing or invalid values"
        )
        raise err
    except Exception as err:
        logging.error("Testing encoder_helper: Function failed with error: %s", err)
        raise err


def test_perform_feature_engineering(data):
    """
    Test perform_feature_engineering functionality
    """
    try:
        # Define feature columns
        keep_cols = [
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
        ]

        X_train, X_test, y_train, y_test = perform_feature_engineering(data, keep_cols)

        # Check if splits are non-empty and have correct shapes
        assert len(X_train) > 0
        assert len(X_test) > 0
        assert len(y_train) > 0
        assert len(y_test) > 0
        assert X_train.shape[1] == len(keep_cols)

        logging.info(
            "Testing perform_feature_engineering: SUCCESS - "
            "Train/Test split created with expected shapes"
        )

        return X_train, X_test, y_train, y_test

    except AssertionError as err:
        logging.error(
            "Testing perform_feature_engineering: Data splits have unexpected shapes"
        )
        raise err
    except Exception as err:
        logging.error(
            "Testing perform_feature_engineering: Function failed with error: %s", err
        )
        raise err


def test_train_models(X_train, X_test, y_train, y_test):
    """
    Test train_models functionality
    """
    try:
        # Train models
        rfc_model, lr_model, *predictions = train_models(
            X_train, X_test, y_train, y_test
        )

        # Check if model files exist
        assert os.path.exists("./models/rfc_model.pkl")
        assert os.path.exists("./models/logistic_model.pkl")

        # Check if result plots exist
        assert os.path.exists("./images/results/roc_curve.png")
        assert os.path.exists("./images/results/shap_importance.png")

        logging.info(
            "Testing train_models: SUCCESS - Models trained and saved successfully"
        )

        return rfc_model, lr_model, predictions

    except AssertionError as err:
        logging.error("Testing train_models: Missing expected output files")
        raise err
    except Exception as err:
        logging.error("Testing train_models: Function failed with error: %s", err)
        raise err


def test_classification_report_image(
    y_train, y_test, y_train_preds, y_test_preds, model_name
):
    """
    Test classification_report_image functionality
    """
    try:
        # Generate classification reports
        classification_report_image(
            y_train, y_test, y_train_preds, y_test_preds, model_name
        )

        # Check if report images exist
        assert os.path.exists("./images/results/random_forest_results.png")

        logging.info("Testing classification_report_image: SUCCESS")

    except AssertionError as err:
        logging.error(
            "Testing classification_report_image: Missing expected report images"
        )
        raise err
    except Exception as err:
        logging.error(
            "Testing classification_report_image: Function failed with error: %s", err
        )
        raise err


def test_feature_importance_plot(model, X_data):
    """
    Test feature_importance_plot functionality
    """
    try:
        # Generate feature importance plot
        feature_importance_plot(model, X_data)

        # Check if feature importance plot exists
        assert os.path.exists("./images/results/feature_importances.png")

        logging.info("Testing feature_importance_plot: SUCCESS")

    except AssertionError as err:
        logging.error(
            "Testing feature_importance_plot: Missing feature importance plot"
        )
        raise err
    except Exception as err:
        logging.error(
            "Testing feature_importance_plot: Function failed with error: %s", err
        )
        raise err


if __name__ == "__main__":
    # Create directories if they don't exist
    for dir_path in ["./logs", "./images/eda", "./images/results", "./models"]:
        os.makedirs(dir_path, exist_ok=True)

    # Run all tests in sequence, passing data between them
    df = test_import()
    test_eda(df)
    df_encoded = test_encoder_helper(df)
    X_train, X_test, y_train, y_test = test_perform_feature_engineering(df_encoded)
    (
        rfc_model,
        lr_model,
        (y_train_preds_lr, y_test_preds_lr, y_train_preds_rf, y_test_preds_rf),
    ) = test_train_models(X_train, X_test, y_train, y_test)

    # Test classification reports for both models
    test_classification_report_image(
        y_train, y_test, y_train_preds_lr, y_test_preds_lr, "Logistic Regression"
    )
    test_classification_report_image(
        y_train, y_test, y_train_preds_rf, y_test_preds_rf, "Random Forest"
    )

    # Test feature importance plot
    test_feature_importance_plot(rfc_model, pd.concat([X_train, X_test]))

    logging.info("All tests completed successfully!")
