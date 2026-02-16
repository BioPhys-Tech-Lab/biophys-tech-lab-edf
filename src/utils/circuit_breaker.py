from enum import Enum
import time
from typing import Callable, Any

class CircuitState(Enum):
    CLOSED = "closed" # Normal operation
    OPEN = "open" # Failing, reject
    HALF_OPEN = "half_open" # Testing recovery

class CircuitBreaker:
    """Circuit breaker con fallback automatico"""
    
    def __init__(self, 
                 failure_threshold: int = 5,
                 recovery_timeout: int = 60,
                 expected_exception: Exception = Exception):
        """
        Args:
            failure_threshold: Numero de fallos antes de abrir
            recovery_timeout: Segundos antes de intentar recuperacion
            expected_exception: Tipo de excepcion a capturar
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        
        self.failure_count = 0
        self.last_failure_time = None
        self.state = CircuitState.CLOSED

    def call(self, func: Callable, *args, fallback: Callable = None, **kwargs) -> Any:
        """
        Ejecuta funcion con proteccion de circuit breaker
        
        Args:
            func: Funcion a ejecutar
            fallback: Funcion alternativa si esta abierto
        """
        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitState.HALF_OPEN
                print(f"[CircuitBreaker] Intentando recuperacion")
            else:
                if fallback:
                    print(f"[CircuitBreaker] OPEN - usando fallback")
                    return fallback(*args, **kwargs)
                raise Exception("Circuit breaker is OPEN")

        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except self.expected_exception as e:
            self._on_failure()
            if fallback:
                print(f"[CircuitBreaker] Fallback despues de fallo: {str(e)}")
                return fallback(*args, **kwargs)
            raise

    def _on_success(self):
        """Registra exito y resetea counters"""
        self.failure_count = 0
        self.state = CircuitState.CLOSED

    def _on_failure(self):
        """Registra fallo y evalua apertura"""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN
            print(f"[CircuitBreaker] OPENED despues de {self.failure_count} fallos")

    def _should_attempt_reset(self) -> bool:
        """Evalua si es hora de intentar recuperacion"""
        if self.last_failure_time is None:
            return False
            
        elapsed = time.time() - self.last_failure_time
        return elapsed >= self.recovery_timeout

class FaultTolerantPipeline:
    """Pipeline con recuperacion automatica y fallbacks"""
    
    def __init__(self):
        self.primary_breaker = CircuitBreaker(
            failure_threshold=3,
            recovery_timeout=60
        )
        self.backup_breaker = CircuitBreaker(
            failure_threshold=5,
            recovery_timeout=120
        )

    def process_with_failover(self, data):
        """Procesa con multiples niveles de failover"""
        
        # Level 1: Intenta primario
        try:
            return self.primary_breaker.call(
                self._primary_processor,
                data,
                fallback=self._backup_processor
            )
        except Exception as e:
            print(f"[Pipeline] Fallo primario: {str(e)}")
            
        # Level 2: Backup explicito (si primary breaker falla sin fallback o relanza)
        return self.backup_breaker.call(
            self._backup_processor,
            data,
            fallback=self._local_cache_fallback
        )

    def _primary_processor(self, data):
        """Procesador primario (e.g., TSDB remoto)"""
        import random
        # Simulacion: puede fallar
        if random.random() < 0.1: # 10% prob. fallo
            raise ConnectionError("TSDB timeout")
        return "processed_primary"

    def _backup_processor(self, data):
        """Procesador backup (e.g., SQLite local)"""
        # Implementacion de fallback
        return "processed_backup"

    def _local_cache_fallback(self, data):
        """Fallback final: cache local"""
        return "processed_local_cache"
