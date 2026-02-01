"""
Test Suite for Recipe High Traffic Classification Project

This module contains comprehensive tests for all project packages including:
- Data validation and cleaning
- Feature engineering
- Statistical analysis
- Model training and evaluation
- Visualization utilities
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from model_development import LogisticRegressionModel, RandomForestModel
from model_evaluation import ModelEvaluator
from utils import DataExplorer, FeatureEngineering, ModelInterpreter, StatisticalTester
from validation import DataValidator
from visualization import DataVisualizer, ModelVisualizer

# ============================================================================
# FIXTURES - Reusable test data
# ============================================================================


@pytest.fixture
def sample_dataframe():
    """Create a sample dataframe for testing."""
    np.random.seed(42)
    data = {
        "recipe": range(1, 101),
        "calories": np.random.uniform(100, 500, 100),
        "carbohydrate": np.random.uniform(10, 100, 100),
        "sugar": np.random.uniform(5, 50, 100),
        "protein": np.random.uniform(5, 50, 100),
        "category": np.random.choice(["Chicken", "Pork", "Vegetable", "Dessert", "Breakfast"], 100),
        "servings": np.random.choice([2, 4, 6, 8], 100),
        "high_traffic": np.random.choice([True, False], 100),
    }
    return pd.DataFrame(data)


@pytest.fixture
def sample_df_with_missing():
    """Create a sample dataframe with missing values."""
    np.random.seed(42)
    data = {
        "recipe": range(1, 51),
        "calories": np.random.uniform(100, 500, 50),
        "carbohydrate": np.random.uniform(10, 100, 50),
        "sugar": np.random.uniform(5, 50, 50),
        "protein": np.random.uniform(5, 50, 50),
        "category": np.random.choice(["Chicken", "Pork", "Vegetable", "Dessert", "Breakfast"], 50),
        "servings": np.random.choice([2, 4, 6, 8], 50),
        "high_traffic": np.random.choice([True, False], 50),
    }
    df = pd.DataFrame(data)
    # Add some missing values
    df.loc[0:5, "calories"] = np.nan
    df.loc[0:5, "carbohydrate"] = np.nan
    df.loc[0:5, "sugar"] = np.nan
    df.loc[0:5, "protein"] = np.nan
    return df


@pytest.fixture
def sample_features_and_target():
    """Create sample features and target for model testing."""
    np.random.seed(42)
    X = pd.DataFrame(
        {
            "calories_log": np.random.normal(5, 1, 100),
            "sugar_log": np.random.normal(3, 1, 100),
            "carbohydrate_log": np.random.normal(4, 1, 100),
            "category_Pork": np.random.choice([0, 1], 100),
            "category_Vegetable": np.random.choice([0, 1], 100),
        }
    )
    y = np.random.choice([True, False], 100)
    return X, pd.Series(y)


# ============================================================================
# VALIDATION TESTS
# ============================================================================


class TestDataValidator:
    """Test suite for DataValidator class."""

    def test_validator_initialization(self, sample_dataframe):
        """Test that DataValidator initializes correctly."""
        validator = DataValidator(sample_dataframe)
        assert validator.df is not None
        assert len(validator.df) == 100

    def test_validate_and_clean(self, sample_dataframe):
        """Test complete validation and cleaning pipeline."""
        validator = DataValidator(sample_dataframe)
        df_cleaned = validator.validate_and_clean()
        assert df_cleaned is not None
        assert isinstance(df_cleaned, pd.DataFrame)
        assert "high_traffic" in df_cleaned.columns

    def test_impute_numerical_values(self, sample_df_with_missing):
        """Test numerical value imputation."""
        validator = DataValidator(sample_df_with_missing)
        validator.validate_and_clean()
        df_imputed = validator.impute_numerical_values(
            num_cols=["calories", "carbohydrate", "sugar", "protein"],
            group_by_col="category",
        )
        # Check that imputation worked
        assert df_imputed["calories"].isnull().sum() == 0
        assert df_imputed["carbohydrate"].isnull().sum() == 0

    def test_detect_outliers(self, sample_dataframe):
        """Test outlier detection."""
        validator = DataValidator(sample_dataframe)
        df_cleaned = validator.validate_and_clean()
        outliers = validator.detect_outliers(
            df_cleaned, num_cols=["calories", "carbohydrate"], threshold=2.0
        )
        assert isinstance(outliers, pd.DataFrame)


# ============================================================================
# UTILS TESTS
# ============================================================================


class TestDataExplorer:
    """Test suite for DataExplorer class."""

    def test_data_explorer_check_integrity(self, sample_dataframe, capsys):
        """Test data integrity checking."""
        explorer = DataExplorer()
        explorer.check_data_integrity(sample_dataframe, id_col="recipe")
        captured = capsys.readouterr()
        assert "Missing Values" in captured.out
        assert "Unique IDs" in captured.out

    def test_validate_categorical_columns(self, sample_dataframe, capsys):
        """Test categorical column validation."""
        explorer = DataExplorer()
        explorer.validate_categorical_columns(sample_dataframe)
        captured = capsys.readouterr()
        assert "category" in captured.out.lower() or "servings" in captured.out.lower()


class TestFeatureEngineering:
    """Test suite for FeatureEngineering class."""

    def test_prepare_features_for_modeling(self, sample_dataframe):
        """Test feature preparation with log transformation and encoding."""
        fe = FeatureEngineering()
        X = sample_dataframe[["calories", "sugar", "carbohydrate", "category"]].copy()
        X_transformed = fe.prepare_features_for_modeling(
            X, numerical_cols=["calories", "sugar", "carbohydrate"], categorical_cols=["category"]
        )
        # Check that log columns were created
        assert "calories_log" in X_transformed.columns
        assert "sugar_log" in X_transformed.columns
        # Check that one-hot encoding worked
        assert any("category_" in col for col in X_transformed.columns)

    def test_prepare_train_test_datasets(self):
        """Test train-test dataset preparation."""
        fe = FeatureEngineering()
        np.random.seed(42)
        X_train = pd.DataFrame(
            {
                "calories": np.random.rand(100),
                "sugar": np.random.rand(100),
                "calories_log": np.random.rand(100),
                "sugar_log": np.random.rand(100),
                "category_encoded": np.random.rand(100),
            }
        )
        X_test = pd.DataFrame(
            {
                "calories": np.random.rand(20),
                "sugar": np.random.rand(20),
                "calories_log": np.random.rand(20),
                "sugar_log": np.random.rand(20),
                "category_encoded": np.random.rand(20),
            }
        )
        (X_train_without_nutr, X_test_without_nutr, X_train_with_log, X_test_with_log) = (
            fe.prepare_train_test_datasets(
                X_train,
                X_test,
                numerical_cols_raw=["calories", "sugar"],
                numerical_cols_logs=["calories_log", "sugar_log"],
            )
        )
        # Verify shapes are preserved
        assert X_train_without_nutr.shape[0] == X_train.shape[0]
        assert X_test_without_nutr.shape[0] == X_test.shape[0]


class TestStatisticalTester:
    """Test suite for StatisticalTester class."""

    def test_anova_test_multiple(self, sample_dataframe, capsys):
        """Test ANOVA test functionality."""
        tester = StatisticalTester()
        results = tester.anova_test_multiple(
            sample_dataframe, num_cols=["calories", "protein"], cat_col="category"
        )
        assert isinstance(results, dict)
        assert "calories" in results
        assert "F-statistic" in results["calories"]
        captured = capsys.readouterr()
        assert "ANOVA" in captured.out

    def test_chi_squared_test(self, sample_dataframe, capsys):
        """Test chi-squared test functionality."""
        tester = StatisticalTester()
        results = tester.chi_squared_test(sample_dataframe, cat_col="category")
        assert isinstance(results, dict)
        assert "chi2_statistic" in results
        assert "p_value" in results
        captured = capsys.readouterr()
        assert "Chi-squared" in captured.out


class TestModelInterpreter:
    """Test suite for ModelInterpreter class."""

    def test_logistic_regression_table(self, sample_features_and_target):
        """Test logistic regression coefficient extraction."""
        X, y = sample_features_and_target
        model = LogisticRegression()
        model.fit(X, y)

        interpreter = ModelInterpreter()
        feature_names = list(X.columns)
        table = interpreter.logistic_regression_table(model, feature_names)

        assert isinstance(table, pd.DataFrame)
        assert "Feature" in table.columns
        assert "Coefficient" in table.columns
        # Intercept is included as first row, so length = features + 1
        assert len(table) == len(feature_names) + 1
        assert table.iloc[0]["Feature"] == "Intercept"
        assert table.iloc[0]["Feature"] == "Intercept"


# ============================================================================
# MODEL DEVELOPMENT TESTS
# ============================================================================


class TestLogisticRegressionModel:
    """Test suite for LogisticRegressionModel wrapper."""

    def test_model_initialization(self):
        """Test LogisticRegressionModel initialization."""
        model = LogisticRegressionModel()
        assert model.model is not None

    def test_model_fit_and_predict(self, sample_features_and_target):
        """Test model fitting and prediction."""
        X, y = sample_features_and_target
        model = LogisticRegressionModel()
        model.fit(X, y)

        predictions = model.predict(X)
        assert len(predictions) == len(y)
        assert all(pred in [True, False] for pred in predictions)

    def test_model_predict_proba(self, sample_features_and_target):
        """Test probability predictions."""
        X, y = sample_features_and_target
        model = LogisticRegressionModel()
        model.fit(X, y)

        proba = model.predict_proba(X)
        assert proba.shape == (len(y), 2)
        assert np.all((proba >= 0) & (proba <= 1))

    def test_get_coefficients_table(self, sample_features_and_target):
        """Test coefficient extraction."""
        X, y = sample_features_and_target
        model = LogisticRegressionModel()
        model.fit(X, y)

        coef_table = model.get_coefficients_table()
        assert isinstance(coef_table, pd.DataFrame)
        # Intercept is included as first row, so length = features + 1
        assert len(coef_table) == len(X.columns) + 1
        assert coef_table.iloc[0]["Feature"] == "Intercept"


class TestRandomForestModel:
    """Test suite for RandomForestModel wrapper."""

    def test_model_initialization(self):
        """Test RandomForestModel initialization."""
        model = RandomForestModel()
        assert model.model is not None

    def test_model_fit_and_predict(self, sample_features_and_target):
        """Test model fitting and prediction."""
        X, y = sample_features_and_target
        model = RandomForestModel()
        model.fit(X, y)

        predictions = model.predict(X)
        assert len(predictions) == len(y)
        assert all(pred in [True, False] for pred in predictions)

    def test_get_feature_importance(self, sample_features_and_target):
        """Test feature importance extraction."""
        X, y = sample_features_and_target
        model = RandomForestModel()
        model.fit(X, y)

        importance = model.get_feature_importance()
        assert isinstance(importance, pd.DataFrame)
        assert "Feature" in importance.columns
        assert "Importance" in importance.columns


# ============================================================================
# MODEL EVALUATION TESTS
# ============================================================================


class TestModelEvaluator:
    """Test suite for ModelEvaluator class."""

    def test_calculate_metrics(self):
        """Test metrics calculation."""
        y_true = np.array([1, 0, 1, 1, 0, 1, 0, 0])
        y_pred = np.array([1, 0, 1, 0, 0, 1, 0, 1])
        y_proba = np.array([0.9, 0.2, 0.8, 0.4, 0.1, 0.95, 0.3, 0.6])

        evaluator = ModelEvaluator()
        metrics = evaluator.calculate_metrics(y_true, y_pred, y_proba)

        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "roc_auc" in metrics
        assert all(0 <= v <= 1 for v in metrics.values())

    def test_create_evaluation_table(self):
        """Test evaluation table creation."""
        metrics = {"accuracy": 0.85, "precision": 0.82, "recall": 0.88}

        evaluator = ModelEvaluator()
        table = evaluator.create_evaluation_table("Test Model", metrics)

        assert isinstance(table, pd.DataFrame)
        assert table.iloc[0]["Model"] == "Test Model"

    def test_compare_models(self):
        """Test model comparison."""
        results = {
            "Model A": {"accuracy": 0.85, "precision": 0.82},
            "Model B": {"accuracy": 0.88, "precision": 0.85},
        }

        evaluator = ModelEvaluator()
        comparison = evaluator.compare_models(results)

        assert isinstance(comparison, pd.DataFrame)
        assert len(comparison) == 2
        assert "Model" in comparison.columns


# ============================================================================
# VISUALIZATION TESTS
# ============================================================================


class TestDataVisualizer:
    """Test suite for DataVisualizer class."""

    def test_visualizer_initialization(self):
        """Test DataVisualizer initialization."""
        viz = DataVisualizer()
        assert viz is not None

    def test_static_methods_exist(self):
        """Test that visualization methods exist."""
        viz = DataVisualizer()
        assert hasattr(viz, "plot_histogram")
        assert hasattr(viz, "plot_barplot")
        assert hasattr(viz, "plot_confusion")


class TestModelVisualizer:
    """Test suite for ModelVisualizer class."""

    def test_model_visualizer_initialization(self):
        """Test ModelVisualizer initialization."""
        viz = ModelVisualizer()
        assert viz is not None

    def test_static_methods_exist(self):
        """Test that model visualization methods exist."""
        viz = ModelVisualizer()
        assert hasattr(viz, "plot_feature_importance_rf")
        assert hasattr(viz, "plot_logistic_coefficients")


# ============================================================================
# INTEGRATION TESTS
# ============================================================================


class TestIntegration:
    """Integration tests for complete workflows."""

    def test_data_validation_to_imputation_workflow(self, sample_df_with_missing):
        """Test complete data validation and imputation workflow."""
        validator = DataValidator(sample_df_with_missing)
        validator.validate_and_clean()
        df_imputed = validator.impute_numerical_values(
            num_cols=["calories", "carbohydrate", "sugar", "protein"],
            group_by_col="category",
        )

        # Verify workflow completed successfully
        assert df_imputed is not None
        assert df_imputed["calories"].isnull().sum() == 0

    def test_feature_engineering_to_model_workflow(self, sample_dataframe):
        """Test feature engineering to model training workflow."""
        fe = FeatureEngineering()
        X = sample_dataframe[["calories", "sugar", "carbohydrate", "category"]].copy()
        y = sample_dataframe["high_traffic"]

        # Feature engineering
        X_transformed = fe.prepare_features_for_modeling(
            X, numerical_cols=["calories", "sugar", "carbohydrate"], categorical_cols=["category"]
        )

        # Train test split
        from sklearn.model_selection import train_test_split

        X_train, X_test, y_train, y_test = train_test_split(
            X_transformed, y, train_size=0.75, random_state=42
        )

        # Model training
        model = LogisticRegressionModel()
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        assert len(predictions) == len(y_test)

    def test_model_evaluation_workflow(self, sample_features_and_target):
        """Test model evaluation workflow."""
        X, y = sample_features_and_target
        from sklearn.model_selection import train_test_split

        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75, random_state=42)

        # Train model
        model = LogisticRegressionModel()
        model.fit(X_train, y_train)

        # Get predictions
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        # Evaluate
        evaluator = ModelEvaluator()
        metrics = evaluator.calculate_metrics(y_test, y_pred, y_proba)

        assert all(key in metrics for key in ["accuracy", "precision", "recall", "roc_auc"])
        assert all(0 <= v <= 1 for v in metrics.values())


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--cov"])
