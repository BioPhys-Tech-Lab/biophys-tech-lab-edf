import numpy as np
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List
from datetime import datetime

class MarketRegime(Enum):
    NORMAL = "normal"
    ELEVATED = "elevated"
    CRISIS = "crisis"

@dataclass
class RiskMetrics:
    vix_equivalent: float # Volatilidad implicita o ratio vs baseline
    drawdown_current: float # Drawdown actual
    prediction_confidence: float # Confianza del modelo

class SafetyBreakerSystem:
    """Sistema que detiene trading automaticamente ante volatilidad extrema"""
    
    def __init__(self):
        self.regime = MarketRegime.NORMAL
        self.regime_history = []
        self.max_drawdown_threshold = 0.10 # 10% max drawdown
        self.position_multiplier = 1.0
        self.trading_enabled = True
        
        # Thresholds de cambio de regimen (ratio de volatilidad)
        self.regime_thresholds = {
            'normal_to_elevated': 3.0, # 3x volatilidad baseline
            'elevated_to_crisis': 10.0, # 10x volatilidad baseline
            'crisis_recovery': 2.0 # 2x para recuperacion (histeresis)
        }

    def evaluate_market_conditions(self, current_metrics: RiskMetrics) -> Dict:
        """
        Evalua condiciones de mercado y toma decisiones automaticas
        
        Returns:
            {
                'regime': MarketRegime,
                'action': 'continue' | 'reduce_position' | 'halt_trading',
                'position_multiplier': float,
                'reason': str
            }
        """
        
        # Step 1: Deteccion de cambio de regimen
        new_regime = self._classify_regime(current_metrics)
        
        if new_regime != self.regime:
            self.regime = new_regime
            self.regime_history.append({
                'timestamp': datetime.now(),
                'old_regime': self.regime,
                'new_regime': new_regime,
                'trigger_metrics': current_metrics
            })
            print(f"[REGIME CHANGE] {self.regime.name}")

        # Step 2: Evaluacion de metricas de riesgo
        action = 'continue'
        reason = 'Operacion normal'
        
        # Check Drawdown - Hard Stop
        if current_metrics.drawdown_current > self.max_drawdown_threshold:
            self.trading_enabled = False
            action = 'halt_trading'
            reason = f"Drawdown {current_metrics.drawdown_current:.1%} > threshold"
            self.position_multiplier = 0.0
            
        elif self.regime == MarketRegime.CRISIS:
            self.position_multiplier = 0.2 # 20% posicion
            action = 'reduce_position'
            reason = "Regimen de crisis detectado"
            
        elif self.regime == MarketRegime.ELEVATED:
            self.position_multiplier = 0.5 # 50% posicion
            action = 'reduce_position'
            reason = "Volatilidad elevada"
            
        else:
            self.position_multiplier = 1.0
            action = 'continue'

        # Step 3: Validacion de confianza del modelo
        if self.trading_enabled and current_metrics.prediction_confidence < 0.6:
            # Si la confianza es baja, reducimos exposicion aun mas
            action = 'reduce_position' if action != 'halt_trading' else action
            reason = f"Baja confianza: {current_metrics.prediction_confidence:.1%}" if action != 'halt_trading' else reason
            self.position_multiplier *= 0.5 # Reducir a la mitad lo que ya hubiera

        return {
            'regime': self.regime.value,
            'action': action,
            'position_multiplier': self.position_multiplier,
            'reason': reason,
            'trading_enabled': self.trading_enabled
        }

    def _classify_regime(self, metrics: RiskMetrics) -> MarketRegime:
        """Clasifica regimen de mercado basado en volatilidad"""
        
        # VIX-like metric: ratio de volatilidad respecto a baseline
        volatility_ratio = metrics.vix_equivalent
        
        # Histeresis simple: subir es facil, bajar es dificil
        if volatility_ratio > self.regime_thresholds['elevated_to_crisis']:
            return MarketRegime.CRISIS
            
        elif volatility_ratio > self.regime_thresholds['normal_to_elevated']:
            # Si ya estamos en crisis, necesitamos bajar mas para salir (histeresis)
            if self.regime == MarketRegime.CRISIS and volatility_ratio > self.regime_thresholds['crisis_recovery']:
                 return MarketRegime.CRISIS
            return MarketRegime.ELEVATED
            
        else:
            # Si estamos en elevated, necesitamos bajar mas para normalizar
            if self.regime == MarketRegime.ELEVATED and volatility_ratio > 1.5: # Ejemplo histeresis
                return MarketRegime.ELEVATED
            if self.regime == MarketRegime.CRISIS: # De crisis no se pasa a normal directo facilmente
                return MarketRegime.ELEVATED
                
            return MarketRegime.NORMAL

    def get_position_adjustment(self, base_position_size: float) -> float:
        """Ajusta tamano de posicion segun condiciones"""
        if not self.trading_enabled:
            return 0.0
        return base_position_size * self.position_multiplier

    def get_system_status(self) -> Dict:
        """Status completo del sistema"""
        return {
            'current_regime': self.regime.value,
            'position_multiplier': self.position_multiplier,
            'trading_enabled': self.trading_enabled,
            'regime_history_length': len(self.regime_history),
            'recent_transitions': self.regime_history[-5:] if self.regime_history else []
        }