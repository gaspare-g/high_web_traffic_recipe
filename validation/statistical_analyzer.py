"""
Statistical Analysis Module

This module provides statistical testing functions for data analysis and model evaluation.
Includes ANOVA tests, chi-squared tests, and model coefficient extraction utilities.
"""

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, f_oneway


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

        # Verify length matches expected number of features
        expected_len = len(feature_names)
        assert len(all_coefs) == expected_len, (
            f"Length mismatch: {len(all_coefs)} (coefs) != {expected_len} (features)"
        )

        # Create feature list including intercept label
        feature_list = ["Intercept"] + list(feature_names)

        # Create and return formatted dataframe
        return pd.DataFrame({"Feature": feature_list, "Coefficient": all_coefs})
