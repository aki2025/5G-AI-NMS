from flask import Flask, request
import numpy as np
import tensorflow as tf

app = Flask(__name__)

global_model = tf.keras.models.load_model('model_code/traffic_prediction_model.h5')
client_weights = []
client_volumes = []

def select_clients(min_data=100):
    """Select clients with sufficient data."""
    return [i for i, v in enumerate(client_volumes) if v > min_data]

@app.route('/update', methods=['POST'])
def update_model():
    """Aggregate client weights with non-IID handling."""
    data = request.json
    weights = [np.array(w) for w in data['weights']]
    data_volume = data['data_volume']
    client_weights.append(weights)
    client_volumes.append(data_volume)

    if len(client_weights) >= 3:
        selected_indices = select_clients(min_data=100)
        if not selected_indices:
            return {"status": "error", "message": "No valid clients"}, 400
        selected_weights = [client_weights[i] for i in selected_indices]
        selected_volumes = [client_volumes[i] for i in selected_indices]
        total_volume = sum(selected_volumes)
        avg_weights = [np.sum([w[i] * v / total_volume for w, v in zip(selected_weights, selected_volumes)], axis=0) for i in range(len(selected_weights[0]))]
        global_model.set_weights(avg_weights)
        global_model.save('model_code/traffic_prediction_model.h5')
        client_weights.clear()
        client_volumes.clear()
    return {"status": "success"}, 200

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)