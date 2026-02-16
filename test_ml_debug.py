import numpy as np
import sys
import os

# Ensure src can be found
sys.path.append(os.getcwd())

print("Starting manual debug...")

try:
    from src.models.drift import DriftDetector
    print("[PASS] Import DriftDetector")
except Exception as e:
    print(f"[FAIL] Import DriftDetector: {e}")

try:
    from src.models.predictor import LatencyOptimizedPredictor
    print("[PASS] Import LatencyOptimizedPredictor")
except Exception as e:
    print(f"[FAIL] Import LatencyOptimizedPredictor: {e}")

def debug_drift():
    print("\n--- Testing DriftDetector ---")
    try:
        baseline = np.random.randn(1000, 5)
        detector = DriftDetector(baseline, window_size=100)
        print("Initialized DriftDetector")
        
        shifted_data = np.random.randn(100, 5) + 5
        fake_preds = np.random.randn(100)
        
        report = detector.update_batch(shifted_data, fake_preds)
        print(f"Update batch returned status: {report.get('status', 'OK')}")
        
        if 'overall_drift' in report:
            print(f"Drift Score: {report['overall_drift']['overall_score']}")
            print(f"Alert Level: {report['overall_drift']['alert_level']}")
        else:
            print("No drift report generated (insufficient data?)")
            
    except Exception as e:
        print(f"[FAIL] DriftDetector logic: {e}")
        import traceback
        traceback.print_exc()

def debug_predictor():
    print("\n--- Testing Predictor ---")
    try:
        predictor = LatencyOptimizedPredictor(model_path="dummy.pkl")
        print("Initialized Predictor (Mock mode likely)")
        
        features = {
            'price': 45000.0, 'volume': 1000000.0, 'spread': 0.001, 
            'momentum': 0.05, 'volatility': 0.02
        }
        pred = predictor.predict('BTC', features)
        print(f"Prediction: {pred}")
        
        stats = predictor.get_latency_stats()
        print(f"Latency Stats: {stats}")
        
    except Exception as e:
        print(f"[FAIL] Predictor logic: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_drift()
    debug_predictor()
