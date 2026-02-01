"""
Model Evaluation Module

This module provides evaluation metrics and comparison utilities for classification models.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


class ModelEvaluator:
    """
    A class to evaluate and compare classification model performance.

    Provides methods to calculate standard metrics and compare models side-by-side.
    """

    @staticmethod
    def calculate_metrics(
        y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray = None
    ) -> dict:
        """
        Calculate comprehensive classification metrics.

        Args:
            y_true (np.ndarray): True labels
            y_pred (np.ndarray): Predicted labels
            y_proba (np.ndarray): Predicted probabilities for positive class

        Returns:
            dict: Dictionary containing accuracy, precision, recall, and ROC AUC
        """
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred),
            "recall": recall_score(y_true, y_pred),
        }

        # Add ROC AUC if probabilities are provided
        if y_proba is not None:
            metrics["roc_auc"] = roc_auc_score(y_true, y_proba)

        return metrics

    @staticmethod
    def create_evaluation_table(model_name: str, metrics: dict) -> pd.DataFrame:
        """
        Create a formatted dataframe from metrics.

        Args:
            model_name (str): Name of the model
            metrics (dict): Dictionary of metrics

        Returns:
            pd.DataFrame: Formatted metrics dataframe
        """
        return pd.DataFrame(
            {"Model": [model_name], **{key: [f"{value:.4f}"] for key, value in metrics.items()}}
        )

    @staticmethod
    def compare_models(results: dict) -> pd.DataFrame:
        """
        Compare multiple models' performance.

        Args:
            results (dict): Dictionary with model names as keys and metrics as values

        Returns:
            pd.DataFrame: Comparison table of all models
        """
        comparison_list = []
        for model_name, metrics in results.items():
            row = {"Model": model_name}
            row.update(metrics)
            comparison_list.append(row)

        return pd.DataFrame(comparison_list)

    @staticmethod
    def print_metrics_report(model_name: str, metrics_train: dict, metrics_test: dict):
        """
        Print a formatted evaluation report for a model.

        Args:
            model_name (str): Name of the model
            metrics_train (dict): Training set metrics
            metrics_test (dict): Test set metrics
        """
        print(f"\n{'=' * 50}")
        print(f"{model_name} - Evaluation Report")
        print(f"{'=' * 50}")

        print("\nTRAINING SET METRICS:")
        print("-" * 30)
        for metric, value in metrics_train.items():
            print(f"{metric.capitalize():15s}: {value:.4f}")

        print("\nTEST SET METRICS:")
        print("-" * 30)
        for metric, value in metrics_test.items():
            print(f"{metric.capitalize():15s}: {value:.4f}")

        print(f"{'=' * 50}\n")
