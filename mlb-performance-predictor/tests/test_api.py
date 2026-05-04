from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_root() -> None:
    response = client.get("/")
    assert response.status_code == 200
    payload = response.json()
    assert payload["project"] == "MLB Performance Predictor API"


def test_players_returns_list() -> None:
    response = client.get("/players")
    assert response.status_code == 200
    assert isinstance(response.json(), list)


def test_custom_prediction_endpoint() -> None:
    payload = {
        "age": 25,
        "pa": 600,
        "hr": 30,
        "runs": 90,
        "rbi": 95,
        "sb": 10,
        "walk_rate": 0.10,
        "strikeout_rate": 0.22,
        "batting_avg": 0.275,
        "obp": 0.360,
        "slg": 0.500,
        "ops": 0.860,
        "iso": 0.225,
        "babip": 0.310,
        "woba": 0.370,
        "wrc_plus": 135,
        "war": 4.5,
    }

    response = client.post("/predict/custom-player", json=payload)
    assert response.status_code in (200, 404)
    if response.status_code == 200:
        assert "predicted_next_season_war" in response.json()


def test_rankings_returns_list() -> None:
    response = client.get("/rankings/projected-war")
    assert response.status_code in (200, 404)
    if response.status_code == 200:
        assert isinstance(response.json(), list)


def test_missing_player_404() -> None:
    response = client.get("/players/999999999")
    assert response.status_code == 404
