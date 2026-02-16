import pandas as pd
import numpy as np
from typing import Tuple, Dict, List
from datetime import timedelta, datetime
import logging

class DataGapRecoveryEngine:
    """Motor de recuperacion de datos faltantes con multiples estrategias"""

    def __init__(self, recovery_config: Dict):
        """
        Args:
            recovery_config: Config con estrategias por duracion
            {
                'max_forward_fill_minutes': 30,
                'max_interpolate_minutes': 60,
                'external_sources': ['yahoo', 'alpha_vantage'],
                'log_gaps': True
            }
        """
        self.config = recovery_config
        self.logger = logging.getLogger(__name__)
        self.gap_registry = [] # Auditar todos los gaps

    def detect_and_recover_gaps(self, 
                                df: pd.DataFrame, 
                                freq: str = '5min') -> Tuple[pd.DataFrame, Dict]:
        """
        Detecta y recupera gaps en datos historicos
        
        Args:
            df: DataFrame con DatetimeIndex
            freq: Frecuencia esperada (e.g., '5min', '1min')
            
        Returns:
            Tupla (df_recovered, recovery_report)
        """
        
        if df.empty:
            return df, {'status': 'empty_dataframe'}

        self.logger.info(f"Iniciando analisis de gaps en {len(df)} registros")
        
        # Step 1: Generar serie de tiempo esperada
        expected_index = pd.date_range(
            start=df.index.min(),
            end=df.index.max(),
            freq=freq
        )
        
        # Step 2: Detectar gaps
        actual_index = df.index
        missing_timestamps = expected_index.difference(actual_index)
        
        gaps_detected = {
            'num_gaps': len(missing_timestamps),
            # Simple heuristic for missing minutes based on freq string
            'total_missing_minutes': len(missing_timestamps) * self._parse_freq_to_minutes(freq),
            'gap_details': []
        }
        
        if len(missing_timestamps) == 0:
            self.logger.info("No se detectaron gaps")
            return df, {'status': 'no_gaps'}
            
        # Step 3: Agrupar gaps contiguos
        gap_groups = self._group_consecutive_gaps(missing_timestamps, freq)
        
        # Step 4: Aplicar estrategia segun duracion
        df_recovered = df.copy()
        
        # Reindexing introduces NaNs for missing rows
        df_recovered = df_recovered.reindex(expected_index)
        
        for gap_start, gap_timestamps in gap_groups.items():
            gap_duration = len(gap_timestamps) * self._parse_freq_to_minutes(freq)
            
            recovery_info = {
                'gap_start': gap_start,
                'gap_end': gap_timestamps[-1],
                'duration_minutes': gap_duration,
                'strategy': None,
                'success': False
            }
            
            # Seleccionar estrategia basada en duracion
            if gap_duration <= self.config['max_forward_fill_minutes']:
                recovery_info['strategy'] = 'forward_fill'
                df_recovered = self._apply_forward_fill(
                    df_recovered, gap_start, gap_timestamps
                )
                recovery_info['success'] = True
                
            elif gap_duration <= self.config['max_interpolate_minutes']:
                recovery_info['strategy'] = 'linear_interpolation'
                df_recovered = self._apply_interpolation(
                    df_recovered, gap_start, gap_timestamps
                )
                recovery_info['success'] = True
                
            else:
                # Intentar recuperacion desde fuente externa
                recovery_info['strategy'] = 'external_source'
                df_recovered = self._try_external_recovery(
                    df_recovered, gap_start, gap_timestamps
                )
                if df_recovered is not None:
                    recovery_info['success'] = True
                else: 
                    # If external recovery failed or returned None, mark as excluded but keep NaNs or original
                    recovery_info['strategy'] = 'excluded'
                    recovery_info['success'] = False
            
            gaps_detected['gap_details'].append(recovery_info)
            self.gap_registry.append(recovery_info)

        # Step 5: Validar recuperacion
        validation_report = self._validate_recovery(df, df_recovered)
        
        successful_gaps = sum(1 for g in gaps_detected['gap_details'] if g['success'])
        self.logger.info(
            f"Recuperacion completada: {successful_gaps} de {len(gaps_detected['gap_details'])} gaps"
        )
        
        return df_recovered, {
            'gaps_detected': gaps_detected,
            'validation': validation_report
        }

    def _group_consecutive_gaps(self, missing_timestamps, freq) -> Dict:
        """Agrupa timestamps faltantes consecutivos"""
        if len(missing_timestamps) == 0:
            return {}
            
        gap_groups = {}
        # Convert index to list for iteration
        ts_list = missing_timestamps.tolist()
        current_group_start = ts_list[0]
        current_group = [ts_list[0]]
        
        freq_delta = pd.to_timedelta(freq)
        
        for i in range(1, len(ts_list)):
            ts = ts_list[i]
            prev_ts = ts_list[i-1]
            
            # Si esta a 1 periodo de distancia, es parte del mismo gap
            if (ts - prev_ts) == freq_delta:
                current_group.append(ts)
            else:
                # Nuevo gap
                gap_groups[current_group_start] = current_group
                current_group_start = ts
                current_group = [ts]
                
        # Add last group
        gap_groups[current_group_start] = current_group
        return gap_groups

    def _apply_forward_fill(self, df: pd.DataFrame, 
                           gap_start: datetime, 
                           gap_timestamps: list) -> pd.DataFrame:
        """Aplica forward fill para gaps cortos"""
        # Pandas ffill handles this elegantly if data is sorted
        # We limit the fill to the gap area implicitly by reindexing previously
        df.loc[gap_timestamps] = df.loc[gap_timestamps].fillna(method='ffill')
        # If the gap starts at the beginning, ffill might fail, so we might need backfill or just leave NaN
        # But for 'forward fill' strategy strictly, we assume preceding data exists.
        
        # More robust specific fill:
        # Get value before gap
        # loc indexing with slicing requires handling if gap_start is first element
        
        # Simple approach applied to whole DF for safety in this scope:
        # df.fillna(method='ffill', inplace=True) 
        # But this would fill ALL gaps. We only want to fill SPECIFIC gaps.
        
        # Correct block approach:
        # Find index location of gap start
        try:
            start_loc = df.index.get_loc(gap_start)
            if start_loc > 0:
                valid_val = df.iloc[start_loc - 1]
                # Fill only the rows in this gap
                for cols in df.columns:
                    df.loc[gap_timestamps, cols] = valid_val[cols]
        except KeyError:
            pass # Index issues
            
        return df

    def _apply_interpolation(self, df: pd.DataFrame, 
                            gap_start: datetime, 
                            gap_timestamps: list) -> pd.DataFrame:
        """Aplica interpolacion lineal para gaps medianos"""
        # Interpolate everything, but in a real selective engine we would mask others.
        # Here we assume the DF passed is the one being worked on.
        # Since we reindexed, the gap rows are NaNs.
        # We can run interpolate on numeric columns.
        
        df_numeric = df.select_dtypes(include=[np.number])
        # Limit direction='both' or similar could be used
        df[df_numeric.columns] = df_numeric.interpolate(method='linear')
        return df

    def _try_external_recovery(self, df: pd.DataFrame, 
                              gap_start: datetime, 
                              gap_timestamps: list) -> pd.DataFrame:
        """Intenta recuperacion desde fuentes externas"""
        for source in self.config.get('external_sources', []):
            try:
                external_data = self._fetch_from_source(source, gap_timestamps)
                if external_data is not None:
                    df.update(external_data)
                    self.logger.info(f"Gap recuperado desde {source}")
                    return df
            except Exception as e:
                self.logger.warning(f"Fallo recuperacion desde {source}: {str(e)}")
        
        return df # Return original (with NaNs) if all fail

    def _fetch_from_source(self, source: str, timestamps: list):
        """Fetch datos desde fuente externa (stub)"""
        # Implementacion especifica por fuente
        return None

    def _validate_recovery(self, df_original: pd.DataFrame, 
                          df_recovered: pd.DataFrame) -> Dict:
        """Valida que recuperacion no introduzca sesgos"""
        
        # Verificar que no hay mas NaNs
        # Note: df_recovered might still have NaNs if strategies failed (excluded)
        nan_count_after = df_recovered.isna().sum().sum()
        
        # Verificar estadisticas no cambiaron radicalmente
        # Catch empty or all-NaN DFs
        if df_original.empty or df_recovered.empty:
             return {'status': 'empty_comparison'}

        stats_diff = {}
        try:
            # Compare mean of numeric columns
            orig_mean = df_original.mean(numeric_only=True).mean()
            rec_mean = df_recovered.mean(numeric_only=True).mean()
            
            diff = abs(orig_mean - rec_mean)
            stats_diff['mean_diff'] = diff
        except:
            stats_diff['error'] = 'calculation_failed'

        return {
            'nan_count': int(nan_count_after),
            'stats_diff': stats_diff,
            'validation_passed': nan_count_after == 0 # Simplistic pass condition
        }

    def _parse_freq_to_minutes(self, freq: str) -> int:
        """Helper to parse frequency string to integer minutes"""
        # Very basic parser for '5min', '1H', etc.
        try:
            return int(pd.to_timedelta(freq).total_seconds() / 60)
        except:
            return 1 # Fallback