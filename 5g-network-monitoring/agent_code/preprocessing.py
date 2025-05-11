from kafka import KafkaConsumer
from flask import Flask, request
import pandas as pd
import numpy as np
import joblib
import json
import logging
import tensorflow as tf

app = Flask(__name__)
logging.basicConfig(filename='agent_code/agent_log.txt', level=logging.INFO)

MODEL_FEATURE_COLUMNS = [
    "concurrent_users", "ul_traffic", "dl_traffic", "ul_throughput", "dl_throughput",
    "latency_ms", "packet_loss_rate", "rsrp_dbm", "rsrq_db", "sinr_db", "cpu_usage", "mem_usage",
    "urllc_latency", "isac_signal"
]
EXTERNAL_FEATURES = ["temperature", "precipitation", "congestion_score", "event_density", "urban_density", "social_trend_score"]

with open('agent_code/config.json', 'r') as f:
    config = json.load(f)
    mapping = config["mapping"]

fusion_interpreter = tf.lite.Interpreter(model_path='model_code/fusion_model.tflite')
fusion_interpreter.allocate_tensors()
fusion_input_details = fusion_interpreter.get_input_details()
fusion_output_details = fusion_interpreter.get_output_details()

@app.route('/preprocess', methods=['POST'])
def preprocess_log():
    """Preprocess log and apply fusion model."""
    log = request.json
    if not log:
        return {"status": "error", "message": "Empty log"}, 400

    mapped_entry = {standard: log.get(vendor, np.nan) for standard, vendor in mapping.items()}
    df = pd.DataFrame([mapped_entry])
    df['log_time'] = pd.to_datetime(df['log_time'])
    df = df.fillna(0)
    log_features = df[MODEL_FEATURE_COLUMNS].astype(float)

    external = log.get('external', {})
    external_features = [external.get(f, 0.0) for f in EXTERNAL_FEATURES]
    combined_features = np.concatenate([log_features.values[0], external_features])

    scaler = joblib.load('model_code/scaler.joblib')
    scaled_features = scaler.transform([combined_features])[0]

    fusion_interpreter.set_tensor(fusion_input_details[0]['index'], scaled_features.astype(np.float32).reshape(1, -1))
    fusion_interpreter.invoke()
    health_score = float(fusion_interpreter.get_tensor(fusion_output_details[0]['index'])[0])

    unmapped = [col for col in log.keys() if col not in mapping.values() and col != 'external']
    if unmapped:
        logging.info(f"Unmapped columns: {unmapped}")

    return {
        "scaled_features": scaled_features.tolist(),
        "timestamp": df['log_time'].values[0].tolist(),
        "health_score": health_score
    }, 200

if __name__ == "__main__":
    consumer = KafkaConsumer('network_logs', bootstrap_servers=['localhost:9092'], auto_offset_reset='latest')
    for message in consumer:
        log = json.loads(message.value)
        response = requests.post('http://localhost:5003/preprocess', json=log)
        # Handle response
    app.run(host='0.0.0.0', port=5003)