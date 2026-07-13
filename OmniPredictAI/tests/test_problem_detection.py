import unittest

import pandas as pd

from preprocessing import detect_problem_type, train_models, select_best_model


class ProblemDetectionTests(unittest.TestCase):
    def test_detects_classification_for_categorical_target(self):
        target = pd.Series(["yes", "no", "yes", "maybe"])
        self.assertEqual(detect_problem_type(target), "classification")

    def test_detects_regression_for_continuous_numeric_target(self):
        target = pd.Series([1.2, 3.4, 2.7, 5.8])
        self.assertEqual(detect_problem_type(target), "regression")

    def test_trains_classification_models_and_returns_scores(self):
        X = pd.DataFrame({
            "feature_1": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            "feature_2": [1.0, 1.1, 1.2, 1.3, 1.4, 1.5],
        })
        y = pd.Series([0, 0, 0, 1, 1, 1])
        result = train_models(X, y, problem_type="classification")
        self.assertIn("logistic_regression", result["scores"])
        self.assertGreaterEqual(result["scores"]["logistic_regression"], 0)

    def test_selects_best_classification_model(self):
        X = pd.DataFrame({
            "feature_1": [0.0, 0.1, 0.2, 0.3, 1.0, 1.1, 1.2, 1.3],
            "feature_2": [0.0, 0.2, 0.1, 0.3, 1.0, 1.2, 1.1, 1.3],
        })
        y = pd.Series([0, 0, 0, 0, 1, 1, 1, 1])
        result = select_best_model(X, y, problem_type="classification")
        self.assertIn("best_model_name", result)
        self.assertIn("metrics", result)
        self.assertTrue(result["best_model_path"].endswith(".pkl"))


if __name__ == "__main__":
    unittest.main()
