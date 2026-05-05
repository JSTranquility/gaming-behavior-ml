from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "online_gaming_behavior_dataset.csv"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
TARGET = "EngagementLevel"
ID_COLUMN = "PlayerID"
RANDOM_STATE = 42


def build_preprocessor(features: pd.DataFrame) -> ColumnTransformer:
    categorical_columns = features.select_dtypes(include=["object", "string"]).columns.tolist()
    numeric_columns = features.select_dtypes(exclude=["object", "string"]).columns.tolist()

    return ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_columns),
            ("numeric", StandardScaler(), numeric_columns),
        ]
    )


def build_models(preprocessor: ColumnTransformer) -> dict[str, Pipeline]:
    return {
        "logistic_regression": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                (
                    "model",
                    LogisticRegression(
                        max_iter=2000,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "random_forest": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=250,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }


def save_confusion_matrix(y_test: pd.Series, predictions: pd.Series, labels: list[str]) -> None:
    matrix = confusion_matrix(y_test, predictions, labels=labels)
    display = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=labels)
    display.plot(cmap="Blues", values_format="d")
    plt.title("Matriz de confusion - EngagementLevel")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "confusion_matrix.png", dpi=160)
    plt.close()


def save_feature_importance(best_model: Pipeline) -> None:
    preprocessor = best_model.named_steps["preprocessor"]
    model = best_model.named_steps["model"]

    feature_names = preprocessor.get_feature_names_out()
    importances = pd.Series(model.feature_importances_, index=feature_names)
    top_features = importances.sort_values(ascending=False).head(15)

    plt.figure(figsize=(10, 6))
    sns.barplot(x=top_features.values, y=top_features.index, color="#4C78A8")
    plt.title("Variables mas importantes")
    plt.xlabel("Importancia")
    plt.ylabel("Variable")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "feature_importance.png", dpi=160)
    plt.close()

    top_features.to_csv(REPORTS_DIR / "feature_importance.csv", header=["importance"])


def main() -> None:
    MODELS_DIR.mkdir(exist_ok=True)
    REPORTS_DIR.mkdir(exist_ok=True)

    df = pd.read_csv(DATA_PATH)
    features = df.drop(columns=[TARGET, ID_COLUMN])
    target = df[TARGET]

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=target,
    )

    preprocessor = build_preprocessor(features)
    models = build_models(preprocessor)

    results = []
    trained_models = {}

    for model_name, pipeline in models.items():
        pipeline.fit(x_train, y_train)
        predictions = pipeline.predict(x_test)

        accuracy = accuracy_score(y_test, predictions)
        macro_f1 = f1_score(y_test, predictions, average="macro")
        report = classification_report(y_test, predictions, digits=4)

        results.append(
            {
                "model": model_name,
                "accuracy": accuracy,
                "macro_f1": macro_f1,
            }
        )
        trained_models[model_name] = pipeline

        (REPORTS_DIR / f"{model_name}_classification_report.txt").write_text(
            report,
            encoding="utf-8",
        )

    results_df = pd.DataFrame(results).sort_values("macro_f1", ascending=False)
    results_df.to_csv(REPORTS_DIR / "model_comparison.csv", index=False)

    best_model_name = results_df.iloc[0]["model"]
    best_model = trained_models[best_model_name]
    best_predictions = best_model.predict(x_test)

    labels = sorted(target.unique())
    save_confusion_matrix(y_test, best_predictions, labels)

    if best_model_name == "random_forest":
        save_feature_importance(best_model)

    metadata = {
        "best_model": best_model_name,
        "target": TARGET,
        "features": features.columns.tolist(),
        "labels": labels,
        "accuracy": float(results_df.iloc[0]["accuracy"]),
        "macro_f1": float(results_df.iloc[0]["macro_f1"]),
    }

    joblib.dump(best_model, MODELS_DIR / "engagement_model.joblib")
    joblib.dump(metadata, MODELS_DIR / "engagement_model_metadata.joblib")

    print("Entrenamiento completado.")
    print(results_df.to_string(index=False))
    print(f"Mejor modelo: {best_model_name}")
    print(f"Modelo guardado en: {MODELS_DIR / 'engagement_model.joblib'}")
    print(f"Reportes guardados en: {REPORTS_DIR}")


if __name__ == "__main__":
    main()
