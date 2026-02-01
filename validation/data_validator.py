"""
Data Validation Module

This module handles data validation and cleaning operations for the recipe traffic classification project.
It provides classes and functions to validate, clean, and prepare data for analysis.
"""

import re

import numpy as np
import pandas as pd
from scipy.stats import zscore


class DataValidator:
    """
    A class to validate and clean recipe traffic data.

    This class performs comprehensive validation of the dataset including:
    - Detection and handling of missing values
    - Categorical variable validation
    - Numerical data type conversion and cleaning
    - Outlier detection and analysis
    """

    def __init__(self, df: pd.DataFrame):
        """
        Initialize the DataValidator with a dataframe.

        Args:
            df (pd.DataFrame): The input dataframe to validate and clean
        """
        self.df = df.copy()
        self.df_cleaned = None

    def extract_number(self, text):
        """
        Extract the first number from a text string.

        Args:
            text (str or int): Text to extract number from

        Returns:
            int or None: Extracted number or None if not found
        """
        match = re.search(r"\d+", str(text))
        return int(match.group()) if match else None

    def validate_and_clean(self):
        """
        Perform complete validation and cleaning pipeline.

        Returns:
            pd.DataFrame: Cleaned dataframe with all validations applied
        """
        # Start with a copy
        self.df_cleaned = self.df.copy()

        # Step 1: Clean category column - standardize "Chicken Breast" to "Chicken"
        self._clean_categories()

        # Step 2: Clean servings column - extract numbers from text
        self._clean_servings()

        # Step 3: Recode target variable to boolean
        self._recode_target()

        return self.df_cleaned

    def _clean_categories(self):
        """Clean and standardize category values."""
        # Replace "Chicken Breast" with "Chicken"
        self.df_cleaned["category"] = self.df_cleaned["category"].replace(
            {"Chicken Breast": "Chicken"}
        )
        # Convert to string type for consistency
        self.df_cleaned["category"] = self.df_cleaned["category"].astype("string")

    def _clean_servings(self):
        """Extract numeric values from servings column."""
        # Apply extraction to handle text entries like "4 as a snack"
        self.df_cleaned["servings_clean"] = self.df_cleaned["servings"].apply(self.extract_number)
        # Replace original column and drop temporary column
        self.df_cleaned["servings"] = self.df_cleaned["servings_clean"]
        self.df_cleaned.drop(columns=["servings_clean"], inplace=True)
        # Convert to Int64 type to handle any remaining NaN values
        self.df_cleaned["servings"] = pd.to_numeric(
            self.df_cleaned["servings"], errors="coerce"
        ).astype("Int64")

    def _recode_target(self):
        """
        Recode target variable to boolean.

        Convert "High" to True, and NaN/missing values to False.
        """
        self.df_cleaned["high_traffic"] = (
            self.df_cleaned["high_traffic"].astype(str).str.strip().str.lower() == "high"
        )

    def impute_numerical_values(
        self, num_cols: list, group_by_col: str = "category"
    ) -> pd.DataFrame:
        """
        Impute missing numerical values using group-based mean strategy.

        Args:
            num_cols (list): List of numerical column names to impute
            group_by_col (str): Column to group by for imputation calculation

        Returns:
            pd.DataFrame: Dataframe with imputed values
        """
        df_imputed = self.df_cleaned.copy()

        # Use groupby transform to fill with group mean
        for col in num_cols:
            df_imputed[col] = df_imputed.groupby([group_by_col])[col].transform(
                lambda x: x.fillna(x.mean())
            )
            # Round to 2 decimal places for consistency
            df_imputed[col] = df_imputed[col].round(2)

        return df_imputed

    def detect_outliers(
        self,
        df: pd.DataFrame,
        num_cols: list,
        group_by_col: str = "category",
        threshold: float = 3.0,
    ) -> pd.DataFrame:
        """
        Detect outliers using z-score method within groups.

        Args:
            df (pd.DataFrame): Dataframe to analyze
            num_cols (list): List of numerical columns to check
            group_by_col (str): Column to group by for z-score calculation
            threshold (float): Z-score threshold for outlier detection

        Returns:
            pd.DataFrame: Dataframe containing only outliers
        """
        # Calculate z-scores within groups
        for col in num_cols:
            df[f"{col}_zscore"] = df.groupby(group_by_col)[col].transform(zscore)

        # Get outliers for each column
        outliers_list = []
        for col in num_cols:
            outliers = df.loc[np.abs(df[f"{col}_zscore"]) > threshold]
            outliers_list.append(outliers)

        # Combine all outliers
        if outliers_list:
            all_outliers = pd.concat(outliers_list).drop_duplicates()
        else:
            all_outliers = df.iloc[0:0]  # Empty dataframe with same structure

        return all_outliers
