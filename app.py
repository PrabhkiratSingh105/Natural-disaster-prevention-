import os
import sys
from pathlib import Path

from flask import Flask, jsonify, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)


@app.get("/health")
def health():
    return jsonify({"status": "ok"})


@app.get("/api/cyclones")
def cyclones():
    return jsonify([
        {"id": "FANI", "name": "Fani", "center": {"lat": 19.78, "lon": 85.81}},
        {"id": "BIPARJOY", "name": "Biparjoy", "center": {"lat": 23.21, "lon": 68.61}},
        {"id": "TAUKTAE", "name": "Tauktae", "center": {"lat": 20.82, "lon": 71.03}},
    ])


@app.post("/api/predict")
def predict():
    payload = request.get_json(silent=True) or {}
    cyclone = payload.get("cyclone")
    center = payload.get("center")
    if not cyclone or not center:
        return jsonify({"error": "Provide a cyclone and center."}), 400

    try:
        prediction_dir = Path(__file__).resolve().parent / "Prediction-api"
        if str(prediction_dir) not in sys.path:
            sys.path.insert(0, str(prediction_dir))
        import importlib.util

        test_path = prediction_dir / "test.py"
        spec = importlib.util.spec_from_file_location("prediction_test", test_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load prediction module: {test_path}")

        prediction_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(prediction_module)
        predict_for_area = prediction_module.predict_for_area

        center = {**center, "cyclone": cyclone}
        return jsonify(predict_for_area(payload.get("extreme_points"), center))
    except Exception as error:
        app.logger.exception("Cyclone prediction failed")
        status_code = 503 if isinstance(error, FileNotFoundError) else 500
        return jsonify({"error": str(error)}), status_code


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", "5000")),
        debug=True,
    )
