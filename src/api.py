import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request, Security, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from prometheus_client import Counter, Histogram, generate_latest
from pydantic import BaseModel, Field
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from src.config import ROOT, settings

logger = logging.getLogger("gaming_ml_api")

request_counter = Counter(
    "http_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "status"],
)
prediction_counter = Counter(
    "predictions_total",
    "Total predictions made",
    ["prediction_class"],
)
prediction_histogram = Histogram(
    "prediction_duration_seconds",
    "Time spent making predictions",
)


class PlayerRequest(BaseModel):
    Age: int = Field(
        ...,
        ge=10,
        le=100,
        description="Edad del jugador (10-100 años)",
        examples=[22],
    )
    Gender: Literal["Male", "Female", "Non-binary", "Other"] = Field(
        ...,
        description="Género del jugador",
        examples=["Female"],
    )
    Location: Literal["USA", "Europe", "Asia", "Other"] = Field(
        ...,
        description="Ubicación geográfica del jugador",
        examples=["USA"],
    )
    GameGenre: Literal["Action", "RPG", "Simulation", "Sports", "Strategy"] = Field(
        ...,
        description="Género de juego preferido",
        examples=["RPG"],
    )
    PlayTimeHours: float = Field(
        ...,
        ge=0,
        description="Total de horas jugadas",
        examples=[14.5],
    )
    InGamePurchases: bool = Field(
        ...,
        description="Ha realizado compras dentro del juego",
        examples=[True],
    )
    GameDifficulty: Literal["Easy", "Medium", "Hard"] = Field(
        ...,
        description="Nivel de dificultad del juego",
        examples=["Hard"],
    )
    SessionsPerWeek: int = Field(
        ...,
        ge=0,
        description="Número de sesiones de juego por semana",
        examples=[15],
    )
    AvgSessionDurationMinutes: int = Field(
        ...,
        ge=0,
        description="Duración promedio de sesión en minutos",
        examples=[135],
    )
    PlayerLevel: int = Field(
        ...,
        ge=1,
        description="Nivel actual del jugador",
        examples=[72],
    )
    AchievementsUnlocked: int = Field(
        ...,
        ge=0,
        description="Cantidad de logros desbloqueados",
        examples=[38],
    )

    def to_model_input(self, metadata: dict) -> pd.DataFrame:
        data = self.model_dump()
        data["InGamePurchases"] = int(data["InGamePurchases"])
        return pd.DataFrame([data], columns=metadata["features"])


class BatchPlayerRequest(BaseModel):
    players: list[PlayerRequest] = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Lista de jugadores para predicción batch (máx 100)",
    )


class PredictionResponse(BaseModel):
    prediction: str = Field(
        ...,
        description="Nivel de engagement predicho (Low, Medium, High)",
    )
    probabilities: dict[str, float] = Field(
        ...,
        description="Probabilidades para cada nivel de engagement",
    )


class BatchPredictionResponse(BaseModel):
    predictions: list[PredictionResponse] = Field(
        ...,
        description="Lista de predicciones para cada jugador",
    )
    count: int = Field(
        ...,
        description="Número total de predicciones realizadas",
    )


class ErrorResponse(BaseModel):
    error: str = Field(..., description="Descripción del error")
    detail: str = Field(default="", description="Detalles adicionales del error")
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="Timestamp del error",
    )


class HealthResponse(BaseModel):
    status: str
    model: str
    timestamp: str
    model_loaded: bool
    features_count: int
    uptime_seconds: float | None = None


app = FastAPI(
    title="Gaming Engagement Prediction API",
    description="""
    ## API de Predicción de Engagement para Jugadores Online
    
    Esta API permite predecir el nivel de engagement de jugadores basándose en su comportamiento y características demográficas.
    
    ### Características:
    - **Predicción de Engagement**: Clasifica jugadores en niveles Low, Medium, High
    - **Modelo ML**: Utiliza Random Forest entrenado con datos reales
    - **Precisión**: ~90% accuracy en el conjunto de prueba
    - **Variables Consideradas**: Sesiones, tiempo de juego, nivel del jugador, logros, etc.
    - **Batch Predictions**: Permite predecir múltiples jugadores en una sola petición
    - **Rate Limiting**: Protección contra abuso con límites por IP
    - **API Key Auth**: Autenticación opcional con API keys
    - **Prometheus Metrics**: Métricas de rendimiento expuestas para monitoreo
    
    ### Uso:
    1. Envía datos del jugador al endpoint `/predict`
    2. Recibe predicción y probabilidades para cada nivel
    3. Integra fácilmente con aplicaciones web o móviles
    
    ### Tecnologías:
    - FastAPI para el servidor API
    - Scikit-learn para el modelo de Machine Learning
    - Pandas para procesamiento de datos
    """,
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    contact={
        "name": "Gaming Behavior ML Team",
        "email": "support@gaming-ml.com",
    },
    license_info={
        "name": "MIT License",
        "url": "https://opensource.org/licenses/MIT",
    },
)

WEB_DIR = ROOT / "web"

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=settings.allow_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

model = None
metadata = None
start_time = None


def get_api_key(request: Request) -> str | None:
    valid_keys = settings.valid_api_keys
    if not valid_keys:
        return None
    auth_header = request.headers.get("X-API-Key")
    if not auth_header or auth_header not in valid_keys:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key. Provide a valid key via X-API-Key header.",
        )
    return auth_header


@app.on_event("startup")
def load_model_and_setup():
    global model, metadata, start_time
    logger.info("Loading ML model from %s", settings.model_path)
    if not settings.model_path.exists():
        raise RuntimeError(
            f"Model file not found at {settings.model_path}. Run training first."
        )
    if not settings.metadata_path.exists():
        raise RuntimeError(
            f"Metadata file not found at {settings.metadata_path}. Run training first."
        )
    model = joblib.load(settings.model_path)
    metadata = joblib.load(settings.metadata_path)
    start_time = time.time()
    logger.info(
        "Model loaded successfully: %s (accuracy=%.4f)",
        metadata["best_model"],
        metadata["accuracy"],
    )


@app.exception_handler(RateLimitExceeded)
async def rate_limit_exceeded_handler(request: Request, exc: RateLimitExceeded):
    request_counter.labels(
        method=request.method,
        endpoint=request.url.path,
        status="429",
    ).inc()
    return JSONResponse(
        status_code=429,
        content=ErrorResponse(
            error="Rate limit exceeded",
            detail=str(exc),
        ).model_dump(),
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled exception: %s", exc)
    request_counter.labels(
        method=request.method,
        endpoint=request.url.path,
        status="500",
    ).inc()
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal server error",
            detail="An unexpected error occurred. Please try again later.",
        ).model_dump(),
    )


@app.get(
    "/",
    summary="Frontend de la aplicacion",
    description="Sirve la interfaz web de la aplicacion",
    tags=["General"],
)
def root() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


@app.get(
    "/api/info",
    summary="Información de la API",
    description="Retorna información general sobre la API y endpoints disponibles",
    tags=["General"],
)
def api_info() -> dict:
    return {
        "message": "Gaming Engagement Prediction API",
        "description": "API para predecir el nivel de engagement de jugadores online",
        "version": "2.0.0",
        "endpoints": {
            "docs": "/docs - Documentación interactiva Swagger UI",
            "redoc": "/redoc - Documentación alternativa ReDoc",
            "health": "/health - Verificar estado del servicio",
            "metrics": "/metrics - Métricas de Prometheus",
            "predict": "POST /predict - Realizar predicción individual",
            "predict_batch": "POST /predict/batch - Predicción batch (hasta 100 jugadores)",
        },
        "model_info": {
            "type": "Random Forest Classifier",
            "accuracy": "~90%",
            "classes": ["Low", "Medium", "High"],
        },
    }


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Verificar estado del servicio",
    description="Endpoint para monitoreo y verificación de que la API está funcionando correctamente",
    tags=["General"],
)
def health() -> HealthResponse:
    uptime = None
    if start_time is not None:
        uptime = time.time() - start_time
    return HealthResponse(
        status="ok",
        model=metadata["best_model"] if metadata else "not_loaded",
        timestamp=datetime.now(timezone.utc).isoformat(),
        model_loaded=model is not None,
        features_count=len(metadata["features"]) if metadata else 0,
        uptime_seconds=round(uptime, 2) if uptime is not None else None,
    )


@app.get(
    "/metrics",
    summary="Métricas de Prometheus",
    description="Expone métricas de rendimiento para scraping con Prometheus",
    tags=["Monitoring"],
)
def metrics():
    return generate_latest()


@app.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Predecir engagement del jugador",
    description="""
    Realiza la predicción del nivel de engagement de un jugador basándose en sus características.
    
    Este endpoint utiliza el modelo de Random Forest entrenado para clasificar a los jugadores
    en tres niveles de engagement: Low, Medium, High.
    """,
    tags=["Predicción"],
)
@limiter.limit(settings.rate_limit)
def predict_engagement(
    player: PlayerRequest,
    request: Request,
    api_key: str | None = Security(get_api_key),
) -> PredictionResponse:
    start = time.perf_counter()
    logger.info("Prediction request received for age=%d genre=%s", player.Age, player.GameGenre)

    player_df = player.to_model_input(metadata)
    prediction = model.predict(player_df)[0]
    probabilities = model.predict_proba(player_df)[0]

    probability_by_class = {
        label: round(float(prob), 4)
        for label, prob in zip(model.classes_, probabilities, strict=True)
    }

    elapsed = time.perf_counter() - start
    prediction_histogram.observe(elapsed)
    prediction_counter.labels(prediction_class=str(prediction)).inc()

    logger.info(
        "Prediction completed: class=%s latency=%.3fs",
        prediction,
        elapsed,
    )

    return PredictionResponse(
        prediction=prediction,
        probabilities=probability_by_class,
    )


@app.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
    summary="Predecir engagement de múltiples jugadores",
    description="""
    Realiza predicciones para múltiples jugadores en una sola petición (máximo 100).
    
    Ideal para análisis de cohortes o procesamiento por lotes.
    """,
    tags=["Predicción"],
    responses={
        200: {
            "description": "Predicciones batch realizadas exitosamente",
            "content": {
                "application/json": {
                    "example": {
                        "predictions": [
                            {
                                "prediction": "High",
                                "probabilities": {"High": 0.9, "Low": 0.028, "Medium": 0.072},
                            },
                            {
                                "prediction": "Medium",
                                "probabilities": {"High": 0.15, "Low": 0.25, "Medium": 0.6},
                            },
                        ],
                        "count": 2,
                    }
                }
            },
        },
    },
)
@limiter.limit(settings.rate_limit)
def predict_engagement_batch(
    batch: BatchPlayerRequest,
    request: Request,
    api_key: str | None = Security(get_api_key),
) -> BatchPredictionResponse:
    start = time.perf_counter()
    logger.info("Batch prediction request received for %d players", len(batch.players))

    player_dfs = [player.to_model_input(metadata) for player in batch.players]
    combined_df = pd.concat(player_dfs, ignore_index=True)

    predictions = model.predict(combined_df)
    probabilities = model.predict_proba(combined_df)

    results = []
    for i, pred in enumerate(predictions):
        prob_by_class = {
            label: round(float(prob), 4)
            for label, prob in zip(model.classes_, probabilities[i], strict=True)
        }
        results.append(PredictionResponse(prediction=pred, probabilities=prob_by_class))
        prediction_counter.labels(prediction_class=str(pred)).inc()

    elapsed = time.perf_counter() - start
    prediction_histogram.observe(elapsed)

    logger.info(
        "Batch prediction completed: count=%d latency=%.3fs",
        len(results),
        elapsed,
    )

    return BatchPredictionResponse(predictions=results, count=len(results))


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    elapsed = time.perf_counter() - start

    request_counter.labels(
        method=request.method,
        endpoint=request.url.path,
        status=str(response.status_code),
    ).inc()

    logger.info(
        "%s %s %d %.3fs",
        request.method,
        request.url.path,
        response.status_code,
        elapsed,
    )

    return response
