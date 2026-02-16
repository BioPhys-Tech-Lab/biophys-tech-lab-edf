from pydantic import BaseModel, Field, validator, root_validator
from datetime import datetime
from typing import Optional, List
import numpy as np

class AssetPriceData(BaseModel):
    """Validacion de datos de entrada para prediccion"""
    asset_id: str = Field(..., min_length=1, max_length=10)
    current_price: float = Field(..., gt=0, lt=1e6, description="Precio actual del activo")
    volume: float = Field(..., ge=0, le=1e9, description="Volumen de trading")
    bid_ask_spread: float = Field(..., ge=0, le=0.1, description="Diferencia bid-ask")
    momentum_24h: float = Field(..., ge=-1, le=1, description="Momentum ultimas 24h")
    volatility: float = Field(..., ge=0, le=2, description="Volatilidad diaria")
    timestamp: datetime = Field(default_factory=datetime.now)

    @validator('current_price')
    def validate_price_not_extreme(cls, v, values):
        """Previene saltos de precio anomalos (>500%)"""
        # Nota: En un caso real, 'prev_price' vendria de contexto externo o del mismo payload si fuera batch.
        # Aqui simulamos la logica si existiera ese campo, o simplemente validamos rangos absolutos si no.
        if 'prev_price' in values and values['prev_price'] > 0:
            change_pct = abs(v - values['prev_price']) / values['prev_price']
            if change_pct > 5.0: # 500% cambio
                raise ValueError(f"Cambio de precio sospechoso: {change_pct*100:.1f}%")
        return v

    @validator('volume')
    def validate_volume(cls, v):
        """Rechaza volumenes nulos o cero"""
        if v == 0:
            raise ValueError("Volumen no puede ser cero")
        return v

    @root_validator(skip_on_failure=True)
    def validate_consistency(cls, values):
        """Validacion de consistencia entre campos"""
        if values.get('bid_ask_spread') is not None and values.get('bid_ask_spread') < 0:
             raise ValueError("Bid-ask spread no puede ser negativo")
        return values

    class Config:
        schema_extra = {
            "example": {
                "asset_id": "BTC",
                "current_price": 45000.50,
                "volume": 1000000,
                "bid_ask_spread": 0.001,
                "momentum_24h": 0.05,
                "volatility": 0.02
            }
        }

class PredictionRequest(BaseModel):
    """Request para prediccion con metadatos"""
    data: AssetPriceData
    request_id: Optional[str] = None
    include_confidence: bool = True

class PredictionResponse(BaseModel):
    """Response de prediccion con auditoria completa"""
    request_id: str
    prediction: float
    confidence_interval: Optional[tuple] = None
    model_version: str
    latency_ms: float
    timestamp: datetime
    status: str = "success"

class ErrorResponse(BaseModel):
    """Respuesta de error estandarizada"""
    error_code: str
    error_message: str
    timestamp: datetime
    request_id: Optional[str] = None