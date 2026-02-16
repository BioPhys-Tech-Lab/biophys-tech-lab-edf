import numpy as np
from scipy.stats import ks_2samp, wasserstein_distance
from typing import Tuple, Dict
import logging
from datetime import datetime

class DriftDetector:
    """Detector de drift multi-metrica con alertas automaticas"""

    def __init__(self, baseline_data: np.ndarray, window_size: int = 1000):
        """
        Args:
            baseline_data: Datos de entrenamiento para establecer baseline
            window_size: Numero de predicciones para ventana movil
        """
        self.baseline_features = baseline_data
        self.window_size = window_size
        self.feature_buffer = []
        self.prediction_buffer = []
        self.label_buffer = []
        
        self.drift_history = []
        self.logger = self._setup_logger()

        # Estadisticas baseline
        self.baseline_stats = self._compute_baseline_stats(baseline_data)

        # Thresholds para alertas
        self.drift_thresholds = {
            'ks_statistic': 0.15, # KS test
            'wasserstein': 0.3, # Wasserstein distance
            'total_variation': 0.2, # TVD
            'performance_degradation': 0.05 # 5% drop
        }

    def _setup_logger(self):
        logger = logging.getLogger('DriftDetector')
        logger.setLevel(logging.DEBUG)
        return logger

    def _compute_baseline_stats(self, data: np.ndarray) -> Dict:
        """Computa estadisticas baseline"""
        return {
            'mean': np.mean(data, axis=0),
            'std': np.std(data, axis=0),
            'percentiles': {
                '25': np.percentile(data, 25, axis=0),
                '50': np.percentile(data, 50, axis=0),
                '75': np.percentile(data, 75, axis=0)
            },
            'distribution': self._estimate_density(data)
        }

    @staticmethod
    def _estimate_density(data: np.ndarray):
        """Estima densidad de probabilidad"""
        return {
            'method': 'kde_bandwidth_scott',
            'bins': 50
        }

    def update_batch(self, 
                     features: np.ndarray, 
                     predictions: np.ndarray, 
                     labels: np.ndarray = None) -> Dict:
        """Actualiza buffers y detecta drift"""
        
        # Agregar a buffers
        # Asegurarse de que features sea lista de listas o array 2D si viene un solo sample
        if features.ndim == 1:
            features = features.reshape(1, -1)
        self.feature_buffer.extend(features)
        self.prediction_buffer.extend(predictions)
        
        if labels is not None:
            self.label_buffer.extend(labels)

        # Mantener ventana de tamano fijo
        if len(self.feature_buffer) > self.window_size:
            self.feature_buffer = self.feature_buffer[-self.window_size:]
            self.prediction_buffer = self.prediction_buffer[-self.window_size:]
            if self.label_buffer:
                self.label_buffer = self.label_buffer[-self.window_size:]

        # Ejecutar deteccion de drift
        # Solo ejecutar si tenemos suficientes datos en la ventana (e.g. 50%)
        if len(self.feature_buffer) >= self.window_size // 2:
            drift_report = self._detect_drifts()
            
            # Generar alertas si necesario
            self._generate_alerts(drift_report)
            
            return drift_report
            
        return {'status': 'insufficient_data'}

    def _detect_drifts(self) -> Dict:
        """Ejecuta suite completa de deteccion de drift"""
        current_data = np.array(self.feature_buffer)
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'window_size': len(self.feature_buffer),
            'drifts': {}
        }

        # 1. Prueba Kolmogorov-Smirnov
        ks_stats = []
        # Iterar sobre features. Baseline debe tener mismo dim.
        num_features = self.baseline_features.shape[1]
        
        for feature_idx in range(num_features):
            # Asegurarse que current data tenga esas columnas
            if feature_idx < current_data.shape[1]:
                stat, p_value = ks_2samp(
                    self.baseline_features[:, feature_idx],
                    current_data[:, feature_idx]
                )
                ks_stats.append({
                    'feature': feature_idx,
                    'statistic': float(stat),
                    'p_value': float(p_value),
                    'drifted': stat > self.drift_thresholds['ks_statistic']
                })

        report['drifts']['ks_test'] = {
            'metrics': ks_stats,
            'num_drifted_features': sum(1 for m in ks_stats if m['drifted']),
            'drift_severity': float(np.mean([m['statistic'] for m in ks_stats])) if ks_stats else 0.0
        }

        # 2. Distancia Wasserstein
        wasserstein_dists = []
        for feature_idx in range(num_features):
             if feature_idx < current_data.shape[1]:
                dist = wasserstein_distance(
                    self.baseline_features[:, feature_idx],
                    current_data[:, feature_idx]
                )
                wasserstein_dists.append({
                    'feature': feature_idx,
                    'distance': float(dist),
                    'drifted': dist > self.drift_thresholds['wasserstein']
                })

        report['drifts']['wasserstein'] = {
            'metrics': wasserstein_dists,
            'num_drifted_features': sum(1 for m in wasserstein_dists if m['drifted']),
            'total_distance': float(np.mean([m['distance'] for m in wasserstein_dists])) if wasserstein_dists else 0.0
        }

        # 3. Concept Drift (si labels disponibles)
        if len(self.label_buffer) > 0:
            concept_drift = self._detect_concept_drift()
            report['drifts']['concept'] = concept_drift

        # 4. Performance degradation
        if len(self.prediction_buffer) > 100 and len(self.label_buffer) > 100:
            perf_drift = self._measure_performance_drift()
            report['drifts']['performance'] = perf_drift

        # Determinar drift global
        report['overall_drift'] = self._compute_overall_drift_score(report)
        
        return report

    def _detect_concept_drift(self) -> Dict:
        """Detecta cambios en P(Y|X) usando tecnicas supervisadas"""
        # Implementacion simplificada (Placeholder del PDF)
        return {
            'method': 'supervised_concept_drift',
            'detected': False,
            'severity': 0.0
        }

    def _measure_performance_drift(self) -> Dict:
        """Compara performance actual vs baseline"""
        current_predictions = np.array(self.prediction_buffer)
        
        # Debemos asegurar que label_buffer tenga mismo tamano para comparar
        min_len = min(len(self.label_buffer), len(current_predictions))
        current_labels = np.array(self.label_buffer[-min_len:])
        current_preds = current_predictions[-min_len:]
        
        current_mae = np.mean(np.abs(current_labels - current_preds))
        baseline_mae = 0.02 # Establecido en entrenamiento
        
        degradation_rate = (current_mae - baseline_mae) / baseline_mae if baseline_mae > 0 else 0
        
        return {
            'baseline_mae': baseline_mae,
            'current_mae': float(current_mae),
            'degradation_rate': float(degradation_rate),
            'drifted': degradation_rate > self.drift_thresholds['performance_degradation']
        }

    def _compute_overall_drift_score(self, report: Dict) -> Dict:
        """Score agregado de drift"""
        scores = [
            report['drifts']['ks_test']['drift_severity'],
            report['drifts']['wasserstein']['total_distance'] / 10, # Normalizar
        ]
        
        if 'performance' in report['drifts']:
            scores.append(
                report['drifts']['performance']['degradation_rate']
            )
            
        mean_score = np.mean(scores) if scores else 0.0
        
        alert_level = 'NORMAL'
        if mean_score > 0.3:
            alert_level = 'CRITICAL'
        elif mean_score > 0.15:
            alert_level = 'WARNING'
            
        return {
            'overall_score': float(mean_score),
            'alert_level': alert_level,
            'requires_retraining': mean_score > 0.2
        }

    def _generate_alerts(self, report: Dict):
        """Genera alertas basadas en drift"""
        alert_level = report['overall_drift']['alert_level']
        
        if alert_level != 'NORMAL':
            self.logger.warning(
                f"DRIFT ALERT [{alert_level}]: Overall score = "
                f"{report['overall_drift']['overall_score']:.4f}"
            )
            
        if report['overall_drift']['requires_retraining']:
            self.logger.critical("RETRAINING REQUIRED")
            
        self.drift_history.append({
            'timestamp': report['timestamp'],
            'alert_level': alert_level,
            'overall_score': report['overall_drift']['overall_score']
        })

    def get_drift_report_summary(self) -> str:
        """Reporte en formato legible"""
        recent = self.drift_history[-10:] if self.drift_history else []
        return f"Ultimas 10 detecciones: {recent}"