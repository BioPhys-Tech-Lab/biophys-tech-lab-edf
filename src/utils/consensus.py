import numpy as np
from typing import Dict, List
from scipy.stats import median_abs_deviation as mad

class MultiProviderPriceConsensus:
    """Reconcilia precios de multiples proveedores"""

    def __init__(self, providers: List[str]):
        """
        Args:
            providers: Lista de IDs de proveedores
        """
        self.providers = providers
        self.provider_stats = {
            p: {'errors': 0, 'total': 0, 'reliability': 1.0}
            for p in providers
        }
        self.consensus_history = []

    def reconcile_price(self,
                        prices: Dict[str, float],
                        timestamp: str) -> Dict:
        """
        Reconcilia precios de multiples proveedores

        Args:
            prices: {'provider1': 100.5, 'provider2': 100.45, ...}
        
        Returns:
            {
               'consensus_price': 100.5,
               'method': 'weighted_median',
               'confidence': 0.95,
               ...
            }
        """
        
        # Step 1: Validacion inicial
        valid_prices = {p: v for p, v in prices.items() if v > 0}
        
        if len(valid_prices) == 0:
            return self._handle_no_valid_prices()
            
        if len(valid_prices) == 1:
            provider, price = list(valid_prices.items())[0]
            return {
                'consensus_price': price,
                'method': 'single_provider',
                'confidence': 0.5, # Baja confianza con un proveedor
                'outliers': [],
                'reasoning': f'Solo {provider} tiene precio valido'
            }

        # Step 2: Deteccion de outliers con MAD (Median Absolute Deviation)
        price_values = np.array(list(valid_prices.values()))
        median_price = np.median(price_values)
        deviations = np.abs(price_values - median_price)
        mad_value = mad(price_values) if len(price_values) > 1 else 0
        
        # Threshold: |precio - mediana| > 3*MAD
        # Evitar division por cero si todos los precios son iguales (mad=0)
        outlier_threshold = 3 * mad_value if mad_value > 1e-9 else float('inf')
        
        outliers = []
        inliers = {}
        
        for provider, price in valid_prices.items():
            deviation = abs(price - median_price)
            
            if deviation > outlier_threshold and len(valid_prices) > 2 and mad_value > 1e-9:
                outliers.append(provider)
            else:
                inliers[provider] = price

        # Step 3: Calculo de consenso
        if len(inliers) == 0:
            # Todos son outliers (raro), usar mediana completa
            consensus_price = median_price
            method = 'median_all'
        else:
            # Usar weighted median de inliers
            inlier_prices = np.array(list(inliers.values()))
            inlier_providers = list(inliers.keys())
            
            # Pesos basados en confiabilidad historica
            weights = np.array([
                self.provider_stats[p]['reliability']
                for p in inlier_providers
            ])
            
            # Normalizar pesos
            if weights.sum() == 0: 
                weights = np.ones_like(weights)
            weights = weights / weights.sum()
            
            # Weighted median
            consensus_price = self._weighted_median(inlier_prices, weights)
            method = 'weighted_median_inliers'

        # Step 4: Calculo de confianza
        confidence = self._calculate_confidence(
            valid_prices, consensus_price, outliers
        )

        # Step 5: Actualizar estadisticas de proveedores
        self._update_provider_stats(valid_prices, consensus_price, outliers)

        result = {
            'consensus_price': float(consensus_price),
            'method': method,
            'confidence': float(confidence),
            'outliers': outliers,
            'outlier_reasons': [
                f"{p}: {abs(valid_prices[p] - median_price)/median_price*100:.2f}% del mediana"
                for p in outliers
            ],
            'provider_stats': {
                p: self.provider_stats[p]
                for p in self.providers
            }
        }
        
        self.consensus_history.append({
            'timestamp': timestamp,
            'result': result
        })
        
        return result

    @staticmethod
    def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
        """Calcula mediana ponderada"""
        sorted_indices = np.argsort(values)
        sorted_values = values[sorted_indices]
        sorted_weights = weights[sorted_indices]
        
        cumsum_weights = np.cumsum(sorted_weights)
        target_weight = 0.5
        
        # Encontrar el indice donde el peso acumulado cruza el 50%
        idx = np.argmax(cumsum_weights >= target_weight)
        return sorted_values[idx]

    def _calculate_confidence(self, 
                              prices: Dict[str, float], 
                              consensus: float,
                              outliers: List[str]) -> float:
        """Calcula confianza en consenso"""
        
        # Menos outliers = mas confianza
        total_valid = len(prices)
        if total_valid == 0: return 0.0
        
        outlier_ratio = len(outliers) / total_valid
        
        # Menos dispersion = mas confianza
        price_values = np.array(list(prices.values()))
        mean_val = np.mean(price_values)
        if mean_val == 0: return 0.0
        
        coefficient_variation = np.std(price_values) / mean_val
        
        # Combinar factores. Coeficiente de variacion > 0.1 castiga mucho
        # Limitar max castigo por dispersion
        dispersion_penalty = min(1.0, coefficient_variation / 0.1) 
        
        confidence = (1 - outlier_ratio) * (1 - dispersion_penalty)
        return max(0.1, min(1.0, confidence))

    def _update_provider_stats(self, 
                               prices: Dict[str, float], 
                               consensus: float,
                               outliers: List[str]):
        """Actualiza confiabilidad de proveedores"""
        
        for provider, price in prices.items():
            self.provider_stats[provider]['total'] += 1
            
            if provider in outliers:
                self.provider_stats[provider]['errors'] += 1
                
            # Confiabilidad = (1 - error_rate)
            total = max(self.provider_stats[provider]['total'], 1)
            error_rate = self.provider_stats[provider]['errors'] / total
            
            self.provider_stats[provider]['reliability'] = 1 - min(error_rate, 1.0)

    def _handle_no_valid_prices(self) -> Dict:
        """Maneja caso sin precios validos"""
        return {
            'consensus_price': None,
            'method': 'no_valid_data',
            'confidence': 0.0,
            'outliers': list(self.providers),
            'reasoning': 'Todos los proveedores retornaron precios invalidos'
        }