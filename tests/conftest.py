import sys

import pytest


@pytest.fixture(scope="session", autouse=True)
def setup_mock_model_session():
    """Set up mock model before the api module is imported."""
    from unittest.mock import MagicMock, patch

    mock_model = MagicMock()
    mock_metadata = {
        "best_model": "random_forest",
        "target": "EngagementLevel",
        "features": [
            "Age", "Gender", "Location", "GameGenre", "PlayTimeHours",
            "InGamePurchases", "GameDifficulty", "SessionsPerWeek",
            "AvgSessionDurationMinutes", "PlayerLevel", "AchievementsUnlocked",
        ],
        "labels": ["Low", "Medium", "High"],
        "accuracy": 0.8976,
        "macro_f1": 0.8925,
    }

    def predict_side_effect(df):
        import pandas as pd

        n = len(df)
        return ["High"] * n

    def predict_proba_side_effect(df):
        import pandas as pd

        n = len(df)
        return [[0.028, 0.072, 0.9]] * n

    mock_model.predict.side_effect = predict_side_effect
    mock_model.predict_proba.side_effect = predict_proba_side_effect
    mock_model.classes_ = ["Low", "Medium", "High"]

    def joblib_load_side_effect(path):
        if "model" in str(path) and "metadata" not in str(path):
            return mock_model
        return mock_metadata

    with patch("src.api.joblib.load", side_effect=joblib_load_side_effect):
        with patch("src.config.MODEL_PATH", MagicMock(exists=MagicMock(return_value=True))):
            with patch("src.config.METADATA_PATH", MagicMock(exists=MagicMock(return_value=True))):

                from src.api import app, model, metadata, start_time
                import src.api

                src.api.model = mock_model
                src.api.metadata = mock_metadata
                src.api.start_time = 1000.0

                yield app, mock_model, mock_metadata
