from flask import Flask, request
import numpy as np
import joblib
import logging
from model_code.rl_agent import DQNAgent

app = Flask(__name__)
logging.basicConfig(filename='agent_code/agent_log.txt', level=logging.INFO)

MODEL_FEATURE_COLUMNS = [
    "concurrent_users", "ul_traffic", "dl_traffic", "ul_throughput", "dl_throughput",
    "latency_ms", "packet_loss_rate", "rsrp_dbm", "rsrq_db", "sinr_db", "cpu_usage", "mem_usage",
    "urllc_latency", "isac_signal"
]
EXTERNAL_FEATURES = ["temperature", "precipitation", "congestion_score", "event_density", "urban_density", "social_trend_score"]

ACTIONS = [
    "Increase bandwidth allocation",
    "Reroute traffic due to weather",
    "Increase capacity for event",
    "Investigate fault",
    "No action needed"
]

rl_agent = DQNAgent(state_dim=len(MODEL_FEATURE_COLUMNS + EXTERNAL_FEATURES) + 3, action_dim=len(ACTIONS))
rl_agent.model = joblib.load('model_code/rl_policy.pkl')

@app.route('/suggest', methods=['POST'])
def suggest_actions():
    """Suggest actions using RL and SHAP."""
    data = request.json
    prediction = data['prediction']
    external = data['external']
    latest_feature = np.array(data['latest_feature']).reshape(1, -1)
    health_score = prediction['health_score']

    state = np.concatenate([
        latest_feature[0],
        [prediction['traffic'], 1 if prediction['fault'] else 0, health_score]
    ])
    action_idx = rl_agent.act(state)
    action = ACTIONS[action_idx]

    if prediction['fault']:
        explainer = joblib.load('model_code/fault_explainer.pkl')
        classifier = joblib.load('model_code/fault_classifier_model.pkl')
        shap_values = explainer.shap_values(latest_feature)[1]
        top_feature_idx = np.argmax(np.abs(shap_values[0]))
        top_feature = MODEL_FEATURE_COLUMNS[top_feature_idx]
        action = f"Investigate {top_feature} for {prediction['fault_type']} fault"

    reward = 1.0 if health_score > 0.8 else -1.0  # Placeholder reward
    next_state = state  # Placeholder for next state
    rl_agent.train(state, action_idx, reward, next_state, False)

    logging.info(f"Suggested action: {action}")
    return {"action": action}, 200

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5005)