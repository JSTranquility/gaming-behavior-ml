import pytest

from fastapi.testclient import TestClient
from fastapi import status


@pytest.fixture()
def client(setup_mock_model_session):
    app, _, _ = setup_mock_model_session
    return TestClient(app)


VALID_PLAYER = {
    "Age": 22,
    "Gender": "Female",
    "Location": "USA",
    "GameGenre": "RPG",
    "PlayTimeHours": 14.5,
    "InGamePurchases": True,
    "GameDifficulty": "Hard",
    "SessionsPerWeek": 15,
    "AvgSessionDurationMinutes": 135,
    "PlayerLevel": 72,
    "AchievementsUnlocked": 38,
}


class TestRoot:
    def test_root_returns_html(self, client):
        resp = client.get("/")
        assert resp.status_code == status.HTTP_200_OK
        assert "text/html" in resp.headers["content-type"]

    def test_api_info_returns_json(self, client):
        resp = client.get("/api/info")
        assert resp.status_code == status.HTTP_200_OK
        data = resp.json()
        assert "version" in data
        assert "endpoints" in data
        assert "model_info" in data
        assert data["version"] == "2.0.0"


class TestHealth:
    def test_health_returns_ok(self, client):
        resp = client.get("/health")
        assert resp.status_code == status.HTTP_200_OK
        data = resp.json()
        assert data["status"] == "ok"
        assert data["model_loaded"] is True
        assert data["model"] == "random_forest"
        assert data["features_count"] == 11

    def test_health_has_timestamp(self, client):
        resp = client.get("/health")
        data = resp.json()
        assert "timestamp" in data
        from datetime import datetime

        datetime.fromisoformat(data["timestamp"])

    def test_health_has_uptime(self, client):
        resp = client.get("/health")
        data = resp.json()
        assert data["uptime_seconds"] is not None
        assert isinstance(data["uptime_seconds"], float)


class TestMetrics:
    def test_metrics_returns_prometheus_format(self, client):
        resp = client.get("/metrics")
        assert resp.status_code == status.HTTP_200_OK
        assert "http_requests_total" in resp.text


class TestPredict:
    def test_predict_valid_player(self, client):
        resp = client.post("/predict", json=VALID_PLAYER)
        assert resp.status_code == status.HTTP_200_OK
        data = resp.json()
        assert "prediction" in data
        assert "probabilities" in data
        assert data["prediction"] == "High"
        assert sum(data["probabilities"].values()) == pytest.approx(1.0, abs=0.01)

    def test_predict_with_integer_purchases(self, client):
        player = {**VALID_PLAYER, "InGamePurchases": 1}
        resp = client.post("/predict", json=player)
        assert resp.status_code == status.HTTP_200_OK

    def test_predict_with_boolean_purchases(self, client):
        player = {**VALID_PLAYER, "InGamePurchases": False}
        resp = client.post("/predict", json=player)
        assert resp.status_code == status.HTTP_200_OK

    def test_predict_missing_required_field(self, client):
        player = {**VALID_PLAYER}
        del player["Age"]
        resp = client.post("/predict", json=player)
        assert resp.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_predict_age_too_low(self, client):
        player = {**VALID_PLAYER, "Age": 5}
        resp = client.post("/predict", json=player)
        assert resp.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_predict_age_too_high(self, client):
        player = {**VALID_PLAYER, "Age": 150}
        resp = client.post("/predict", json=player)
        assert resp.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_predict_negative_playtime(self, client):
        player = {**VALID_PLAYER, "PlayTimeHours": -5}
        resp = client.post("/predict", json=player)
        assert resp.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_predict_invalid_gender(self, client):
        player = {**VALID_PLAYER, "Gender": "Unknown"}
        resp = client.post("/predict", json=player)
        assert resp.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_predict_invalid_location(self, client):
        player = {**VALID_PLAYER, "Location": "Mars"}
        resp = client.post("/predict", json=player)
        assert resp.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_predict_invalid_difficulty(self, client):
        player = {**VALID_PLAYER, "GameDifficulty": "Extreme"}
        resp = client.post("/predict", json=player)
        assert resp.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_predict_invalid_genre(self, client):
        player = {**VALID_PLAYER, "GameGenre": "Horror"}
        resp = client.post("/predict", json=player)
        assert resp.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_predict_probability_values(self, client):
        resp = client.post("/predict", json=VALID_PLAYER)
        data = resp.json()
        for key in ["Low", "Medium", "High"]:
            assert key in data["probabilities"]
            assert 0 <= data["probabilities"][key] <= 1


class TestPredictBatch:
    def test_batch_two_players(self, client):
        payload = {
            "players": [
                VALID_PLAYER,
                {**VALID_PLAYER, "Age": 30, "GameGenre": "Action"},
            ]
        }
        resp = client.post("/predict/batch", json=payload)
        assert resp.status_code == status.HTTP_200_OK
        data = resp.json()
        assert data["count"] == 2
        assert len(data["predictions"]) == 2

    def test_batch_empty_list(self, client):
        resp = client.post("/predict/batch", json={"players": []})
        assert resp.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_batch_max_limit(self, client):
        players = [VALID_PLAYER] * 101
        resp = client.post("/predict/batch", json={"players": players})
        assert resp.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_batch_valid_max(self, client):
        players = [VALID_PLAYER] * 100
        resp = client.post("/predict/batch", json={"players": players})
        assert resp.status_code == status.HTTP_200_OK
        assert resp.json()["count"] == 100

    def test_batch_mixed_valid(self, client):
        payload = {
            "players": [
                {"Age": 25, "Gender": "Male", "Location": "Europe", "GameGenre": "Action", "PlayTimeHours": 100, "InGamePurchases": True, "GameDifficulty": "Hard", "SessionsPerWeek": 20, "AvgSessionDurationMinutes": 90, "PlayerLevel": 50, "AchievementsUnlocked": 25},
                {"Age": 18, "Gender": "Female", "Location": "Asia", "GameGenre": "RPG", "PlayTimeHours": 50, "InGamePurchases": False, "GameDifficulty": "Easy", "SessionsPerWeek": 5, "AvgSessionDurationMinutes": 30, "PlayerLevel": 10, "AchievementsUnlocked": 2},
                {"Age": 45, "Gender": "Other", "Location": "USA", "GameGenre": "Simulation", "PlayTimeHours": 10, "InGamePurchases": True, "GameDifficulty": "Medium", "SessionsPerWeek": 3, "AvgSessionDurationMinutes": 45, "PlayerLevel": 5, "AchievementsUnlocked": 1},
            ]
        }
        resp = client.post("/predict/batch", json=payload)
        assert resp.status_code == status.HTTP_200_OK
        assert resp.json()["count"] == 3


class TestGenderOptions:
    def test_nonbinary_gender(self, client):
        player = {**VALID_PLAYER, "Gender": "Non-binary"}
        resp = client.post("/predict", json=player)
        assert resp.status_code == status.HTTP_200_OK

    def test_other_gender(self, client):
        player = {**VALID_PLAYER, "Gender": "Other"}
        resp = client.post("/predict", json=player)
        assert resp.status_code == status.HTTP_200_OK


class TestConfig:
    def test_settings_defaults(self):
        from src.config import settings
        assert settings.host == "0.0.0.0"
        assert settings.port == 8000
        assert settings.log_level == "INFO"

    def test_empty_api_keys(self):
        from src.config import settings
        assert settings.valid_api_keys == set()

    def test_generate_api_key(self):
        from src.config import settings
        key = settings.generate_api_key()
        assert len(key) > 0
        assert isinstance(key, str)
