# Playalytics - Gaming Behavior Machine Learning

Predict player engagement levels using Machine Learning powered by FastAPI and Scikit-learn.

## Features

- **Real-time predictions** — Classify players into Low, Medium, or High engagement
- **Batch predictions** — Predict up to 100 players in a single request
- **~90% accuracy** — Random Forest classifier trained on real gaming behavior data
- **Interactive UI** — Web interface to test predictions without writing code
- **REST API** — Full OpenAPI/Swagger documentation at `/docs`
- **Rate limiting** — Protection against abuse (configurable per-IP limits)
- **API key auth** — Optional authentication via `X-API-Key` header
- **Prometheus metrics** — Monitor performance at `/metrics`
- **CORS configurable** — Works with external frontends and mobile apps

## Quick Start

### 1. Install dependencies

```bash
pip install -r config/requirements.txt
```

### 2. Train the model

```bash
python scripts/main.py --train
```

If `online_gaming_behavior_dataset.csv` is not present in the project root, the training script downloads the dataset from KaggleHub:

```text
rabieelkharoua/predict-online-gaming-behavior-dataset
```

### 3. Start the server

```bash
python scripts/main.py --api
```

The application is now available at **http://127.0.0.1:8000**

- **Web UI** — http://127.0.0.1:8000/
- **API Docs** — http://127.0.0.1:8000/docs
- **Health Check** — http://127.0.0.1:8000/health
- **Metrics** — http://127.0.0.1:8000/metrics

## CLI Commands

| Command | Description |
|---------|-------------|
| `python scripts/main.py` | Interactive menu |
| `python scripts/main.py --train` | Train the model |
| `python scripts/main.py --predict` | Run a sample prediction |
| `python scripts/main.py --pipeline` | Train + predict |
| `python scripts/main.py --api` | Start the API server |

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Web interface (frontend) |
| `GET` | `/api/info` | API metadata |
| `GET` | `/health` | Service health check |
| `GET` | `/docs` | Swagger UI documentation |
| `GET` | `/redoc` | ReDoc documentation |
| `GET` | `/metrics` | Prometheus metrics |
| `POST` | `/predict` | Predict single player engagement |
| `POST` | `/predict/batch` | Predict multiple players (max 100) |

### Example Request

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Age": 22,
    "Gender": "Female",
    "Location": "USA",
    "GameGenre": "RPG",
    "PlayTimeHours": 14.5,
    "InGamePurchases": true,
    "GameDifficulty": "Hard",
    "SessionsPerWeek": 15,
    "AvgSessionDurationMinutes": 135,
    "PlayerLevel": 72,
    "AchievementsUnlocked": 38
  }'
```

### Example Response

```json
{
  "prediction": "High",
  "probabilities": {
    "Low": 0.028,
    "Medium": 0.072,
    "High": 0.9
  }
}
```

## Configuration

Copy `.env.example` to `.env` and adjust settings:

```bash
cp .env.example .env
```

| Variable | Default | Description |
|----------|---------|-------------|
| `GAMING_ML_HOST` | `0.0.0.0` | Server host |
| `GAMING_ML_PORT` | `8000` | Server port |
| `GAMING_ML_CORS_ORIGINS` | `["http://localhost:3000"]` | Allowed CORS origins |
| `GAMING_ML_RATE_LIMIT` | `30/minute` | Rate limit per IP |
| `GAMING_ML_API_KEYS` | *(empty)* | Comma-separated API keys (empty = no auth) |
| `GAMING_ML_LOG_LEVEL` | `INFO` | Logging level |

### Enable API Key Authentication

```bash
# Generate a key
python -c "from src.config import settings; print(settings.generate_api_key())"

# Set in .env
GAMING_ML_API_KEYS="your-generated-key-here"
```

Then include the key in requests:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-key-here" \
  -d '{...}'
```

## Model Performance

| Model | Accuracy | Macro F1 |
|-------|----------|----------|
| Random Forest | **89.76%** | **89.25%** |
| Logistic Regression | 78.59% | 78.71% |

## Project Structure

```
├── config/
│   └── requirements.txt       # Python dependencies
├── src/
│   ├── api.py                 # FastAPI server with all endpoints
│   ├── config.py              # Pydantic settings & environment
│   ├── train_engagement_model.py   # Model training pipeline
│   └── predict_engagement.py       # CLI prediction utility
├── scripts/
│   └── main.py                # Entry point with CLI menu
├── tests/
│   ├── conftest.py            # Test fixtures & mocks
│   └── test_api.py            # 28 API tests
├── models/                    # Serialized model & metadata
├── reports/                   # Training reports & charts
├── web/                       # Frontend HTML
├── .env.example               # Environment template
└── README.md
```

## Running Tests

```bash
python -m pytest tests/ -v
```

## Deployment

Esta API se puede subir como un servicio web Python. Antes de desplegar, asegúrate de tener estos archivos:

```text
models/engagement_model.joblib
models/engagement_model_metadata.joblib
```

La API los carga al iniciar. Si no existen, el servidor fallará durante `startup`.

Puedes generarlos con:

```bash
python scripts/main.py --train
```

Si el CSV no está en el repositorio, el entrenamiento descarga el dataset desde KaggleHub.

### Render

1. Sube el proyecto a GitHub.
2. En Render, crea un **Web Service** desde el repositorio.
3. Usa estos comandos:

```bash
pip install -r requirements.txt
uvicorn src.api:app --host 0.0.0.0 --port $PORT
```

También puedes usar el archivo `render.yaml` incluido para crear el servicio como Blueprint.

Variables recomendadas:

```bash
GAMING_ML_HOST=0.0.0.0
GAMING_ML_LOG_LEVEL=INFO
GAMING_ML_RATE_LIMIT=30/minute
GAMING_ML_API_KEYS=your-production-key
GAMING_ML_CORS_ORIGINS=["https://your-frontend-domain.com"]
```

### Railway / Heroku-style platforms

El `Procfile` incluido expone el proceso web:

```bash
web: uvicorn src.api:app --host 0.0.0.0 --port $PORT
```

### Docker

Puedes construir y ejecutar la API con:

```bash
docker build -t playalytics-api .
docker run -p 8000:8000 playalytics-api
```

Para plataformas que inyectan `PORT`, cambia el comando de arranque a:

```bash
uvicorn src.api:app --host 0.0.0.0 --port $PORT
```

## Tech Stack

- **FastAPI** — Web framework
- **Scikit-learn** — Machine Learning (Random Forest, Logistic Regression)
- **Pydantic** — Data validation & settings management
- **SlowAPI** — Rate limiting
- **Prometheus Client** — Metrics
- **Pandas** — Data processing
- **Pytest** — Testing

## License

MIT
