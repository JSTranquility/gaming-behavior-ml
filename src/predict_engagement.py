from datetime import datetime, timezone

from pathlib import Path

import joblib
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "engagement_model.joblib"
METADATA_PATH = ROOT / "models" / "engagement_model_metadata.joblib"


def main() -> None:
    model = joblib.load(MODEL_PATH)
    metadata = joblib.load(METADATA_PATH)

    sample_player = {
        "Age": 22,
        "Gender": "Female",
        "Location": "USA",
        "GameGenre": "RPG",
        "PlayTimeHours": 14.5,
        "InGamePurchases": 1,
        "GameDifficulty": "Hard",
        "SessionsPerWeek": 15,
        "AvgSessionDurationMinutes": 135,
        "PlayerLevel": 72,
        "AchievementsUnlocked": 38,
    }

    player_df = pd.DataFrame([sample_player], columns=metadata["features"])
    prediction = model.predict(player_df)[0]
    probabilities = model.predict_proba(player_df)[0]
    probability_by_class = dict(zip(model.classes_, probabilities, strict=True))

    print("=" * 50)
    print("GAMING BEHAVIOR - PREDICCION DE ENGAGEMENT")
    print(f"Timestamp: {datetime.now(timezone.utc).isoformat()}")
    print("=" * 50)
    print("\nJugador de ejemplo:")
    for key, value in sample_player.items():
        print(f"  {key}: {value}")

    print(f"\nPrediccion de engagement: {prediction}")
    print("\nProbabilidades:")
    for label, probability in sorted(probability_by_class.items()):
        print(f"  {label}: {probability:.2%}")
    print("=" * 50)


if __name__ == "__main__":
    main()
