import tensorflow as tf
import numpy as np
import requests
from tensorflow_privacy import DPGradientDescentOptimizer

def train_local_model(data, model_path):
    """Train local model with differential privacy."""
    model = tf.keras.models.load_model(model_path)
    optimizer = DPGradientDescentOptimizer(learning_rate=0.01, noise_multiplier=1.0)
    model.compile(optimizer=optimizer, loss='mse')
    X, y = data['features'], data['targets']
    model.fit(X, y, epochs=5, batch_size=32, verbose=0)
    return model.get_weights()

def send_weights_to_server(weights, server_url, client_id, data_volume):
    """Send weights to federated server."""
    try:
        response = requests.post(server_url, json={
            'weights': [w.tolist() for w in weights],
            'client_id': client_id,
            'data_volume': data_volume
        })
        response.raise_for_status()
    except Exception as e:
        print(f"Failed to send weights: {e}")

if __name__ == "__main__":
    data = {'features': np.random.rand(100, len(MODEL_FEATURE_COLUMNS + EXTERNAL_FEATURES)), 'targets': np.random.rand(100, 1)}
    weights = train_local_model(data, 'model_code/traffic_prediction_model.h5')
    send_weights_to_server(weights, 'http://localhost:5000/update', client_id='edge1', data_volume=100)