"""
Utility Module

This module provides utility functions for data exploration, validation output,
and common analysis operations used throughout the notebook workflow.
"""

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, f_oneway


class DataExplorer:
    """
    A class for initial data exploration and validation output.

    Provides utilities for examining datasets, displaying validation results,
    and checking data integrity.
    """

    @staticmethod
    def load_and_examine_data(filepath: str):
        """
        Load CSV data and display initial examination info.

        Args:
            filepath (str): Path to the CSV file

        Returns:
            pd.DataFrame: Loaded dataframe
        """
        df = pd.read_csv(filepath)
        print("=== DataFrame Info ===")
        print(df.info())
        print("\n=== General Stats for All Columns ===")
        print(df.describe(include="all").T)
        return df

    @staticmethod
    def check_data_integrity(
        df: pd.DataFrame, id_col: str = "recipe", target_col: str = "high_traffic"
    ):
        """
        Check for missing values and data integrity issues.

        Args:
            df (pd.DataFrame): Input dataframe
            id_col (str): Name of identifier column
            target_col (str): Name of target column
        """
        print("\n=== Missing Values ===")
        print(df.isnull().sum())

        # Verify identifier uniqueness
        unique_ids = df[id_col].nunique()
        total_ids = len(df[id_col])
        print(f"\nUnique IDs: {unique_ids}, Total: {total_ids}")
        if unique_ids != total_ids:
            print("WARNING: Recipe IDs are not unique!")

        # Examine target variable
        print(f"\nUnique {target_col} values:")
        print(df[target_col].unique())
        print(df[target_col].value_counts())

    @staticmethod
    def validate_categorical_columns(df: pd.DataFrame, expected_categories: set = None):
        """
        Validate and examine categorical columns.

        Args:
            df (pd.DataFrame): Input dataframe
            expected_categories (set): Set of expected category values
        """
        # Examine unique values in category column
        print("\nUnique category values:")
        print(df["category"].unique())
        print(df["category"].value_counts())

        # Compare with expected categories if provided
        if expected_categories:
            actual_categories = set(df["category"].dropna().unique())
            print("\nCategories not in expected set:")
            print(actual_categories - expected_categories)
            print("Categories missing from dataset (but expected):")
            print(expected_categories - actual_categories)

        # Examine servings column
        print("\nUnique servings values:")
        print(df["servings"].unique())
        print(df["servings"].value_counts())

    @staticmethod
    def display_cleaning_results(
        df: pd.DataFrame, target_col: str = "high_traffic", expected_categories: set = None
    ):
        """
        Display validation results after cleaning.

        Args:
            df (pd.DataFrame): Cleaned dataframe
            target_col (str): Name of target column
            expected_categories (set): Set of expected category values
        """
        # Verify category cleaning
        if expected_categories:
            actual_categories = set(df["category"].dropna().unique())
            print("\nCategories not in expected set:")
            print(actual_categories - expected_categories)
            print("Categories missing from dataset (but expected):")
            print(expected_categories - actual_categories)

        # Display servings value counts after cleaning
        print("\nServings value counts after cleaning:")
        print(df["servings"].value_counts().sort_index())

        # Display unique values of target column after recoding
        print(f"\nUnique {target_col} values after recoding:")
        print(df[target_col].unique())
        print(df[target_col].value_counts())

    @staticmethod
    def display_missing_rows(df: pd.DataFrame, num_rows: int = 10):
        """
        Display rows with missing values.

        Args:
            df (pd.DataFrame): Input dataframe
            num_rows (int): Number of rows to display
        """
        df_missing = df[df.isnull().any(axis=1)]
        print("Sample of rows with missing values:")
        print(df_missing.head(num_rows))
        return df_missing


class FeatureEngineering:
    """
    A class for feature engineering and preparation operations.

    Provides utilities for encoding, transformation, and feature creation.
    """

    @staticmethod
    def prepare_features_for_modeling(
        X: pd.DataFrame, numerical_cols: list, categorical_cols: list
    ) -> pd.DataFrame:
        """
        Prepare features for modeling with log transformation and encoding.

        Args:
            X (pd.DataFrame): Input features dataframe with both numerical and categorical
            numerical_cols (list): List of numerical column names
            categorical_cols (list): List of categorical column names

        Returns:
            pd.DataFrame: Transformed features ready for modeling
        """
        X_temp = X.copy()

        # Create log-transformed versions of numerical features
        for col in numerical_cols:
            X_temp[col + "_log"] = np.log1p(X_temp[col])

        # One-hot encode categorical features
        X_transformed = pd.get_dummies(X_temp, columns=categorical_cols, drop_first=False)

        return X_transformed

    @staticmethod
    def prepare_train_test_datasets(
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        numerical_cols_raw: list,
        numerical_cols_logs: list,
    ) -> tuple:
        """
        Create different feature subsets for different models.

        Creates:
        - Dataset without numerical features (category only)
        - Dataset with log-transformed numerical features

        Args:
            X_train (pd.DataFrame): Training features
            X_test (pd.DataFrame): Test features
            numerical_cols_raw (list): Raw numerical column names
            numerical_cols_logs (list): Log-transformed column names

        Returns:
            tuple: (X_train_without_nutr, X_test_without_nutr,
                    X_train_with_log, X_test_with_log)
        """
        # Create datasets without numerical features
        X_train_without_nutr = X_train.drop(columns=numerical_cols_raw + numerical_cols_logs).copy()
        X_test_without_nutr = X_test.drop(columns=numerical_cols_raw + numerical_cols_logs).copy()

        # Create datasets with log-transformed numerical features
        X_train_with_log = X_train.drop(columns=numerical_cols_raw).copy()
        X_test_with_log = X_test.drop(columns=numerical_cols_raw).copy()

        return (X_train_without_nutr, X_test_without_nutr, X_train_with_log, X_test_with_log)


class StatisticalTester:
    """
    A class to perform various statistical tests on data.

    Provides methods for ANOVA tests, chi-squared tests, and other statistical analyses
    relevant to exploratory data analysis and model interpretation.
    """

    @staticmethod
    def anova_test_multiple(df: pd.DataFrame, num_cols: list, cat_col: str = "category"):
        """
        Perform ANOVA tests for multiple numerical columns grouped by categorical column.

        This function tests whether there is a significant difference in numerical
        values across different categories using one-way ANOVA. Results are printed
        with formatting for readability.

        Args:
            df (pd.DataFrame): Input dataframe
            num_cols (list): List of numerical column names to test
            cat_col (str): Categorical column to group by for comparison

        Returns:
            dict: Dictionary with results for each numerical column
        """
        results = {}
        for num_col in num_cols:
            # Create groups for each category value
            groups = [df[num_col][df[cat_col] == group].dropna() for group in df[cat_col].unique()]

            # Perform one-way ANOVA test
            f_stat, p_val = f_oneway(*groups)

            # Format to 3 decimals for readability
            f_stat_fmt = f"{f_stat:.3f}"
            p_val_fmt = f"{p_val:.3f}"
            formatted_cat = cat_col.replace("_", " ").title()

            # Print results
            print(f"ANOVA for '{num_col}' by {formatted_cat}: F = {f_stat_fmt}, p = {p_val_fmt}")

            # Store results
            results[num_col] = {"F-statistic": f_stat_fmt, "p-value": p_val_fmt}

        return results

    @staticmethod
    def chi_squared_test(df: pd.DataFrame, cat_col: str, target_col: str = "high_traffic"):
        """
        Perform chi-squared test of independence between categorical and target variables.

        Tests whether there is a significant association between a categorical feature
        and the target variable using chi-squared test of independence.
        Results are printed in a formatted manner.

        Args:
            df (pd.DataFrame): Input dataframe
            cat_col (str): Categorical column name to test
            target_col (str): Target variable column name

        Returns:
            dict: Dictionary containing chi2 statistic, p-value, and degrees of freedom
        """
        # Create contingency table
        contingency_table = pd.crosstab(df[cat_col], df[target_col])

        # Perform chi-squared test of independence
        chi2, p, dof, expected = chi2_contingency(contingency_table)

        # Print formatted results
        print(f"Chi-squared test between '{cat_col}' and '{target_col}':")
        print(f"Chi2 statistic: {chi2:.3f}")
        print(f"p-value:        {p:.3f}")
        print(f"Degrees of freedom: {dof:.3f}")

        return {
            "chi2_statistic": chi2,
            "p_value": p,
            "degrees_of_freedom": dof,
            "contingency_table": contingency_table,
        }


class ModelInterpreter:
    """
    A class for extracting and interpreting model coefficients and parameters.

    Provides utilities for converting model parameters to readable tables
    and other interpretable formats.
    """

    @staticmethod
    def logistic_regression_table(model, feature_names) -> pd.DataFrame:
        """
        Create a table of logistic regression coefficients.

        Extracts coefficients from a fitted logistic regression model and
        creates a formatted dataframe for interpretation. Includes intercept
        as the first row for complete model representation.

        Args:
            model: Fitted LogisticRegression model from scikit-learn
            feature_names (list): List of feature names (excluding intercept)

        Returns:
            pd.DataFrame: Table with features and their coefficients
                Columns: ['Feature', 'Coefficient']

        Raises:
            AssertionError: If number of coefficients doesn't match feature count
        """
        # Extract coefficients from model
        coef = model.coef_.flatten()
        intercept = model.intercept_.flatten()

        # Combine intercept with coefficients
        all_coefs = np.concatenate([intercept, coef])

        # Verify length matches expected number of features + intercept
        expected_len = len(feature_names) + 1
        assert len(all_coefs) == expected_len, (
            f"Length mismatch: {len(all_coefs)} (coefs) != {expected_len} (features + intercept)"
        )

        # Create feature list including intercept label
        feature_list = ["Intercept"] + list(feature_names)

        # Create and return formatted dataframe
        return pd.DataFrame({"Feature": feature_list, "Coefficient": all_coefs})
