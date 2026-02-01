"""
Model Development Module

This module contains classes and functions for training and comparing classification models
for the recipe traffic prediction task.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


class LogisticRegressionModel:
    """
    Wrapper class for Logistic Regression model for recipe traffic classification.

    This model uses only categorical features (category encoded) to avoid multicollinearity
    issues with numerical features.
    """

    def __init__(
        self,
        solver: str = "lbfgs",
        penalty: str = "l2",
        l1_ratio: float = 0.75,
        C: float = 109.85,
        max_iter: int = 10000,
    ):
        """
        Initialize Logistic Regression model with hyperparameters.

        Args:
            solver (str): Solver algorithm ('lbfgs', 'saga', 'liblinear')
            penalty (str): Penalty type ('l1', 'l2', 'elasticnet', None)
            l1_ratio (float): Mixing parameter for elasticnet (0 to 1)
            C (float): Regularization strength (inverse)
            max_iter (int): Maximum iterations for solver convergence
        """
        self.model = LogisticRegression(
            solver=solver, penalty=penalty, l1_ratio=l1_ratio, C=C, max_iter=max_iter
        )
        self.feature_names = None

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series):
        """
        Fit the Logistic Regression model.

        Args:
            X_train (pd.DataFrame): Training features
            y_train (pd.Series): Training target variable
        """
        self.feature_names = X_train.columns
        self.model.fit(X_train, y_train)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions on new data.

        Args:
            X (pd.DataFrame): Features to predict on

        Returns:
            np.ndarray: Predicted class labels
        """
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Get probability predictions.

        Args:
            X (pd.DataFrame): Features to predict on

        Returns:
            np.ndarray: Probability of each class
        """
        return self.model.predict_proba(X)

    def get_coefficients_table(self) -> pd.DataFrame:
        """
        Get model coefficients as a dataframe.

        Returns:
            pd.DataFrame: Table with feature names and coefficients
        """
        coef = self.model.coef_.flatten()
        intercept = self.model.intercept_.flatten()
        all_coefs = np.concatenate([intercept, coef])
        feature_list = ["Intercept"] + list(self.feature_names)

        return pd.DataFrame({"Feature": feature_list, "Coefficient": all_coefs})


class RandomForestModel:
    """
    Wrapper class for Random Forest Classifier model for recipe traffic classification.

    This model can use both numerical and categorical features and is robust to
    multicollinearity and non-linear relationships.
    """

    def __init__(
        self,
        n_estimators: int = 10,
        max_depth=None,
        min_samples_split: int = 15,
        min_samples_leaf: int = 1,
        max_features: str = "log2",
        bootstrap: bool = True,
        class_weight=None,
    ):
        """
        Initialize Random Forest Classifier with hyperparameters.

        Args:
            n_estimators (int): Number of trees in the forest
            max_depth: Maximum depth of trees (None for unlimited)
            min_samples_split (int): Minimum samples required to split a node
            min_samples_leaf (int): Minimum samples required at a leaf node
            max_features (str): Number of features to consider ('auto', 'sqrt', 'log2')
            bootstrap (bool): Whether to use bootstrap samples
            class_weight: Class weight specification (None or 'balanced')
        """
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            bootstrap=bootstrap,
            class_weight=class_weight,
        )
        self.feature_names = None

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series):
        """
        Fit the Random Forest model.

        Args:
            X_train (pd.DataFrame): Training features
            y_train (pd.Series): Training target variable
        """
        self.feature_names = X_train.columns
        self.model.fit(X_train, y_train)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions on new data.

        Args:
            X (pd.DataFrame): Features to predict on

        Returns:
            np.ndarray: Predicted class labels
        """
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Get probability predictions.

        Args:
            X (pd.DataFrame): Features to predict on

        Returns:
            np.ndarray: Probability of each class
        """
        return self.model.predict_proba(X)

    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importances as a dataframe.

        Returns:
            pd.DataFrame: Sorted feature importances
        """
        importances = self.model.feature_importances_
        importance_df = pd.DataFrame(
            {"Feature": self.feature_names, "Importance": importances}
        ).sort_values(by="Importance", ascending=False)

        return importance_df
