from flask import Flask, request
import tensorflow as tf
import numpy as np
import joblib
from model_code.online_learning import OnlineLearner

app = Flask(__name__)

MODEL_FEATURE_COLUMNS = [
    "concurrent_users", "ul_traffic", "dl_traffic", "ul_throughput", "dl_throughput",
    "latency_ms", "packet_loss_rate", "rsrp_dbm", "rsrq_db", "sinr_db", "cpu_usage", "mem_usage",
    "urllc_latency", "isac_signal"
]
EXTERNAL_FEATURES = ["temperature", "precipitation", "congestion_score", "event_density", "urban_density", "social_trend_score"]

traffic_learner = OnlineLearner('model_code/traffic_prediction_model.h5', len(MODEL_FEATURE_COLUMNS + EXTERNAL_FEATURES))
fault_learner = OnlineLearner('model_code/fault_detection_model.h5', len(MODEL_FEATURE_COLUMNS + EXTERNAL_FEATURES))

@app.route('/predict', methods=['POST'])
def predict():
    """Run predictions with online learning."""
    data = request.json
    feature_history = data['feature_history']
    health_score = data['health_score']
    n = 10
    threshold = 0.05

    if len(feature_history) < 1:
        return {"traffic": 0.0, "fault": False, "fault_type": "none", "health_score": health_score}, 200

    interpreter = tf.lite.Interpreter(model_path='model_code/traffic_prediction_model.tflite')
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    traffic_pred = 0.0
    if len(feature_history) >= n:
        sequence = np.array(feature_history[-n:]).astype(np.float32).reshape((1, n, len(MODEL_FEATURE_COLUMNS + EXTERNAL_FEATURES)))
        interpreter.set_tensor(input_details[0]['index'], sequence)
        interpreter.invoke()
        traffic_pred = float(interpreter.get_tensor(output_details[0]['index'])[0][0])
        traffic_learner.update(sequence[0][-1], traffic_pred)

    fault_interpreter = tf.lite.Interpreter(model_path='model_code/fault_detection_model.tflite')
    fault_interpreter.allocate_tensors()
    fault_input_details = fault_interpreter.get_input_details()
    fault_output_details = fault_interpreter.get_output_details()
    latest_feature = np.array(feature_history[-1]).astype(np.float32).reshape(1, -1)
    fault_interpreter.set_tensor(fault_input_details[0]['index'], latest_feature)
    fault_interpreter.invoke()
    recon = fault_interpreter.get_tensor(fault_output_details[0]['index'])
    mse = np.mean(np.square(latest_feature - recon))
    fault_detected = mse > threshold
    fault_learner.update(latest_feature[0], recon[0])

    fault_type = "none"
    if fault_detected:
        classifier = joblib.load('model_code/fault_classifier_model.pkl')
        fault_type = classifier.predict(latest_feature)[0]

    return {
        "traffic": traffic_pred,
        "fault": fault_detected,
        "fault_type": fault_type,
        "health_score": health_score
    }, 200

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5004)