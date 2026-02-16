from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import time
import logging
import uuid
from datetime import datetime
import sqlite3
import json
import numpy as np

# Imports relativos asumiendo estructura src/
from src.models.validator import (
    PredictionRequest, PredictionResponse, ErrorResponse, 
    AssetPriceData
)
from src.models.predictor import LatencyOptimizedPredictor
from src.models.drift import DriftDetector
from src.utils.circuit_breaker import CircuitBreaker
from src.utils.safety import SafetyBreakerSystem, RiskMetrics as SafetyRiskMetrics

# Configurar logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AppState:
    """Estado global de la aplicacion"""
    def __init__(self):
        self.predictor = None
        self.drift_detector = None
        self.db_connection = None
        self.metrics = {} # Simple in-memory metrics for demo
        self.safety_system = None

app_state = AppState()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle de la aplicacion: startup y shutdown"""
    # Startup
    logger.info("Inicializando servicio de prediccion...")
    
    # 1. Cargar Predictor (Mocked path)
    app_state.predictor = LatencyOptimizedPredictor(
        model_path="models/gradient_boosting_v1.onnx", # Dummy path, class handles mock
        scaler_path="models/scaler.pkl"
    )
    
    # 2. Inicializar Drift Detector
    baseline = app_state.predictor.load_baseline()
    app_state.drift_detector = DriftDetector(
        baseline_data=baseline,
        window_size=100
    )
    
    # 3. Inicializar Safety System
    app_state.safety_system = SafetyBreakerSystem()

    # 4. Conectar DB (SQLite para demo)
    app_state.db_connection = sqlite3.connect(
        "predictions.db", 
        check_same_thread=False
    )
    _init_database(app_state.db_connection)
    
    # 5. Inicializar metricas simples
    app_state.metrics = {
        'total_requests': 0, 
        'sla_violations': 0,
        'drift_alerts': 0
    }

    logger.info("Servicio inicializado correctamente")
    
    yield # Aplicacion corre aqui
    
    # Shutdown
    logger.info("Cerrando servicio...")
    if app_state.db_connection:
        app_state.db_connection.close()
    logger.info("Servicio cerrado")

app = FastAPI(
    title="Predictor de Precios - BioPhys-Tech Lab",
    description="Servicio de prediccion financiera con Gradient Boosting, Blue-Green deployment y Circuit Breakers.",
    version="1.0.0",
    lifespan=lifespan
)

def _init_database(conn: sqlite3.Connection):
    """Inicializa tablas de base de datos"""
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            request_id TEXT UNIQUE,
            asset_id TEXT,
            input_price REAL,
            prediction REAL,
            latency_ms REAL,
            model_version TEXT,
            timestamp TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS drift_alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            alert_level TEXT,
            overall_score REAL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()

@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    """Manejador personalizado de errores de validacion"""
    return JSONResponse(
        status_code=422,
        content=ErrorResponse(
            error_code="VALIDATION_ERROR",
            error_message=str(exc),
            timestamp=datetime.utcnow(),
            request_id=request.headers.get("X-Request-ID")
        ).dict()
    )

@app.get("/health", tags=["Monitoreo"])
async def health_check():
    """Health check endpoint para Kubernetes"""
    status = "healthy"
    if app_state.predictor is None:
        status = "degraded"
    
    return {
        "status": status,
        "timestamp": datetime.utcnow().isoformat(),
        "model_loaded": app_state.predictor is not None
    }

@app.get("/metrics", tags=["Monitoreo"])
async def get_metrics():
    """Metricas de performance del servicio"""
    latency_stats = app_state.predictor.get_latency_stats()
    
    # Agregar safety status
    safety_status = {}
    if app_state.safety_system:
        safety_status = app_state.safety_system.get_system_status()

    return {
        "service_metrics": app_state.metrics,
        "latency_stats": latency_stats,
        "safety_system": safety_status
    }

@app.post("/predict", 
          response_model=PredictionResponse, 
          status_code=200,
          tags=["Predicciones"])
async def predict(
    request: PredictionRequest, 
    background_tasks: BackgroundTasks
):
    """
    Endpoint de prediccion principal
    
    **Latencia SLA**: < 200ms
    **Validacion**: Todos los inputs validados con Pydantic
    **Monitoreo**: Todas las predicciones registradas
    """
    start_time = time.time()
    request_id = request.request_id or str(uuid.uuid4())
    
    # Check Safety Breaker first!
    if app_state.safety_system and not app_state.safety_system.trading_enabled:
        raise HTTPException(
            status_code=503, 
            detail=f"Trading halted by Safety Breaker system. Reason: High Risk Regime detected."
        )

    try:
        # Note: Input validation is handled by Pydantic (PredictionRequest) automatically before entering function.
        logger.debug(f"[{request_id}] Prediccion solicitada para {request.data.asset_id}")
        
        app_state.metrics['total_requests'] += 1

        # Prediccion
        prediction_value = app_state.predictor.predict(
            asset_id=request.data.asset_id,
            features={
                'price': request.data.current_price,
                'volume': request.data.volume,
                'spread': request.data.bid_ask_spread,
                'momentum': request.data.momentum_24h,
                'volatility': request.data.volatility
            }
        )
        
        # Intervalo de confianza
        confidence_interval = app_state.predictor.get_confidence_interval(
            prediction_value
        )
        
        # Calcular latencia
        latency_ms = (time.time() - start_time) * 1000
        
        # Verificar SLA
        if latency_ms > 200:
            logger.warning(f"[{request_id}] SLA violation: {latency_ms:.2f}ms")
            app_state.metrics['sla_violations'] += 1
            
        response = PredictionResponse(
            request_id=request_id,
            prediction=float(prediction_value),
            confidence_interval=confidence_interval if request.include_confidence else None,
            model_version=app_state.predictor.version,
            latency_ms=latency_ms,
            timestamp=datetime.utcnow(),
            status="success"
        )
        
        # Registrar en background (Fire and Forget)
        background_tasks.add_task(
            _log_prediction_task,
            request_id=request_id,
            request_data=request.data,
            response=response
        )
        
        # Actualizar drift detector en background
        background_tasks.add_task(
            _check_drift_task,
            request_id=request_id,
            features=[
                # Must match indices expected by drift detector from baseline
                # Just passing random features matching dimension for this demo integration
                request.data.current_price,
                request.data.volume,
                request.data.bid_ask_spread,
                request.data.momentum_24h,
                request.data.volatility
            ],
            prediction=prediction_value
        )
        
        return response

    except Exception as e:
        logger.exception(f"[{request_id}] Error inesperado: {str(e)}")
        raise HTTPException(status_code=500, detail="Error interno del servidor")

async def _log_prediction_task(request_id: str, request_data: AssetPriceData, response: PredictionResponse):
    """Registra prediccion en BD SQLite"""
    try:
        cursor = app_state.db_connection.cursor()
        cursor.execute("""
            INSERT INTO predictions 
            (request_id, asset_id, input_price, prediction, latency_ms, model_version, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            request_id,
            request_data.asset_id,
            request_data.current_price,
            response.prediction,
            response.latency_ms,
            response.model_version,
            response.timestamp.isoformat()
        ))
        app_state.db_connection.commit()
    except Exception as e:
        logger.error(f"[{request_id}] Error al registrar prediccion: {str(e)}")

async def _check_drift_task(request_id: str, features: list, prediction: float):
    """Detecta drift en background y actualiza Safety System"""
    try:
        drift_report = app_state.drift_detector.update_batch(
            features=np.array([features]),
            predictions=np.array([prediction])
        )
        
        if drift_report.get('status') != 'insufficient_data':
            # Update safety system with latest metrics simulated context
            # In a real system we would calculate drawdown etc.
            
            # Simulate metrics for safety check
            # Extract drift score as proxy for risk if drift is high
            score = 0
            if 'overall_drift' in drift_report and 'overall_score' in drift_report['overall_drift']:
                score = drift_report['overall_drift']['overall_score']
            
            risk_metrics = SafetyRiskMetrics(
                vix_equivalent=1.0 + score, # If drift score high, volatility risk high
                drawdown_current=0.0, # Need external PnL context
                prediction_confidence=0.95
            )
            
            app_state.safety_system.evaluate_market_conditions(risk_metrics)
            
            if drift_report['overall_drift']['alert_level'] != 'NORMAL':
                logger.warning(f"[{request_id}] DRIFT ALERT logged")
                app_state.metrics['drift_alerts'] += 1
                
    except Exception as e:
        logger.error(f"[{request_id}] Error en drift detection: {str(e)}")