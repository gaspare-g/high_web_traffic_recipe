"""
Visualization Module

This module provides functions for visualizing data distributions, relationships,
and model results for the recipe traffic classification project.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix


class DataVisualizer:
    """
    A class to create various visualizations for exploratory and model analysis.

    Provides methods for plotting distributions, relationships, and model results.
    """

    @staticmethod
    def plot_histogram(data: pd.DataFrame, column: str, bins: int = 30):
        """
        Plot histogram with KDE overlay for a numerical column.

        Args:
            data (pd.DataFrame): Input dataframe
            column (str): Column name to plot
            bins (int): Number of bins for histogram
        """
        sns.histplot(data[column], bins=bins, kde=True)
        plt.title(f"Histogram and KDE of {column}")
        plt.xlabel(column)
        plt.ylabel("Frequency")
        plt.show()

    @staticmethod
    def plot_barplot(data: pd.DataFrame, column: str):
        """
        Create horizontal barplot for categorical column.

        Args:
            data (pd.DataFrame): Input dataframe
            column (str): Column name to plot
        """
        col = data[column]
        if col.dtype.name == "category":
            col = col.cat.remove_unused_categories()
        counts = col.value_counts()

        plt.figure(figsize=(10, 6))
        plt.barh(counts.index.astype(str), counts.values, edgecolor="black")
        formatted_column = column.replace("_", " ").title()
        plt.title(f"Barplot of {formatted_column}")
        plt.xlabel("Count")
        plt.ylabel(formatted_column)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_lower_triangle_heatmap(
        df: pd.DataFrame, id_col: str = "recipe", exclude_cols: list = None
    ):
        """
        Plot lower triangle correlation heatmap.

        Args:
            df (pd.DataFrame): Input dataframe
            id_col (str): Identifier column to exclude
            exclude_cols (list): Additional columns to exclude
        """
        if exclude_cols is None:
            exclude_cols = []

        cols = [col for col in df.columns if col != id_col and col not in exclude_cols]
        df_corr = df[cols].copy()

        # Convert categorical columns to numeric
        for col in cols:
            if df_corr[col].dtype == "object":
                df_corr[col] = df_corr[col].astype("category").cat.codes

        corr_matrix = df_corr.corr()

        # Generate mask for upper triangle
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

        plt.figure(figsize=(8, 6))
        sns.heatmap(
            corr_matrix,
            annot=True,
            cmap="coolwarm",
            fmt=".2f",
            mask=mask,
            square=True,
            cbar_kws={"shrink": 0.8},
        )
        plt.title("Lower Triangle Correlation Heatmap")
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_category_high_traffic(
        df: pd.DataFrame, cat_col: str, target_col: str = "high_traffic"
    ):
        """
        Create stacked barplot showing target distribution within categories.

        Args:
            df (pd.DataFrame): Input dataframe
            cat_col (str): Categorical column name
            target_col (str): Target variable column name
        """
        formatted_target = target_col.replace("_", " ").title()
        crosstab = pd.crosstab(df[cat_col], df[target_col], normalize="index")
        crosstab.plot(kind="bar", stacked=True, figsize=(8, 6), colormap="coolwarm")
        plt.ylabel("Proportion")
        plt.title(f"Distribution of {formatted_target} within {cat_col}")
        plt.legend(title=formatted_target)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def draw_boxplot(feature: str, target: str, data: pd.DataFrame):
        """
        Create boxplot comparing feature across target categories.

        Args:
            feature (str): Numerical feature column name
            target (str): Categorical target column name
            data (pd.DataFrame): Input dataframe
        """
        ax = data.boxplot(column=feature, by=target)
        formatted_target = target.replace("_", " ").title()
        plt.title(f"Boxplot of {feature} by {formatted_target}")
        plt.suptitle("")

        labels = data[target].unique()
        ax.set_xticklabels(labels)

        plt.xlabel(formatted_target)
        plt.ylabel(feature)
        plt.show()

    @staticmethod
    def jittered_scatterplot(
        data: pd.DataFrame, num_col: str, target_col: str, jitter: float = 0.05, alpha: float = 0.5
    ):
        """
        Create scatterplot with jitter for binary target visualization.

        Args:
            data (pd.DataFrame): Input dataframe
            num_col (str): Numerical feature column name
            target_col (str): Binary target column name
            jitter (float): Amount of jitter to add
            alpha (float): Transparency level
        """
        x = data[num_col]
        # Add jitter to y for binary variable
        y = data[target_col] + np.random.uniform(-jitter, jitter, size=len(data))
        plt.scatter(x, y, alpha=alpha, s=15)
        plt.xlabel(num_col)
        formatted_target = target_col.replace("_", " ").title()
        plt.ylabel(formatted_target)
        plt.title(f"Scatterplot of {num_col} vs {formatted_target} (with jitter)")
        plt.yticks([0, 1], ["False", "True"])
        plt.show()

    @staticmethod
    def scatterplot_with_binned_means(
        data: pd.DataFrame,
        num_col: str,
        target_col: str,
        bins: int = 5,
        jitter: float = 0.05,
        alpha: float = 0.5,
    ):
        """
        Create scatterplot with binned means overlay.

        Args:
            data (pd.DataFrame): Input dataframe
            num_col (str): Numerical feature column name
            target_col (str): Target column name
            bins (int): Number of bins for means calculation
            jitter (float): Amount of jitter to add
            alpha (float): Transparency level
        """
        DataVisualizer.jittered_scatterplot(data, num_col, target_col, jitter, alpha)

        # Overlay binned means
        data_binned = data.copy()
        data_binned["num_bin"] = pd.cut(data_binned[num_col], bins)
        bin_means = data_binned.groupby("num_bin")[target_col].mean()
        bin_centers = [interval.mid for interval in bin_means.index]
        plt.plot(bin_centers, bin_means.values, color="red", marker="o", linestyle="-")
        plt.show()

    @staticmethod
    def scatterplot_with_reg_line(data: pd.DataFrame, x_col: str, y_col: str):
        """
        Create scatterplot with regression line.

        Args:
            data (pd.DataFrame): Input dataframe
            x_col (str): X-axis column name
            y_col (str): Y-axis column name
        """
        sns.regplot(data=data, x=x_col, y=y_col, ci=None, marker="o", scatter_kws={"s": 25})
        plt.title(f"Scatterplot of {x_col} vs {y_col} with Regression Line")
        plt.show()

    @staticmethod
    def plot_confusion(y_pred: np.ndarray, y_true: np.ndarray, title: str = "Confusion Matrix"):
        """
        Plot confusion matrix visualization.

        Args:
            y_pred (np.ndarray): Predicted values
            y_true (np.ndarray): True values
            title (str): Plot title
        """
        cm = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap="Blues")
        plt.title(title)
        plt.show()


class ModelVisualizer:
    """
    A class for visualizing model-specific results and feature importance.
    """

    @staticmethod
    def plot_feature_importance_rf(
        model, feature_names: list, title: str = "Random Forest Feature Importances"
    ):
        """
        Plot Random Forest feature importances.

        Args:
            model: Fitted Random Forest model
            feature_names (list): List of feature names
            title (str): Plot title
        """
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        sorted_features = [feature_names[i] for i in indices]
        sorted_importances = importances[indices]

        plt.figure(figsize=(10, 6))
        plt.bar(sorted_features, sorted_importances)
        plt.xticks(rotation=45, ha="right")
        plt.ylabel("Importance")
        plt.title(title)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_logistic_coefficients(
        model, feature_names: list, title: str = "Logistic Regression Coefficients"
    ):
        """
        Plot Logistic Regression coefficients.

        Args:
            model: Fitted Logistic Regression model
            feature_names (list): List of feature names
            title (str): Plot title
        """
        coefs = model.coef_[0]
        plt.figure(figsize=(10, 6))
        plt.bar(feature_names, coefs)
        plt.xticks(rotation=45, ha="right")
        plt.ylabel("Coefficient Value")
        plt.title(title)
        plt.tight_layout()
        plt.show()
