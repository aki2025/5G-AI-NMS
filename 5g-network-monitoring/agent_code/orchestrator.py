from flask import Flask, request
import requests
import logging

app = Flask(__name__)
logging.basicConfig(filename='agent_code/agent_log.txt', level=logging.INFO)

ORCHESTRATOR_URL = "https://example.com/onap/api"  # Replace with actual orchestrator URL

@app.route('/execute', methods=['POST'])
def execute_action():
    """Execute action via network orchestrator."""
    action = request.json['action']
    try:
        if "Increase bandwidth" in action:
            response = requests.post(f"{ORCHESTRATOR_URL}/bandwidth", json={"action": "increase"})
        elif "Reroute traffic" in action:
            response = requests.post(f"{ORCHESTRATOR_URL}/reroute", json={"action": "reroute"})
        elif "Increase capacity" in action:
            response = requests.post(f"{ORCHESTRATOR_URL}/capacity", json={"action": "increase"})
        else:
            logging.info(f"Manual action required: {action}")
            return {"status": "manual"}, 200
        response.raise_for_status()
        logging.info(f"Executed action: {action}")
        return {"status": "success"}, 200
    except Exception as e:
        logging.error(f"Failed to execute action: {e}")
        return {"status": "error", "message": str(e)}, 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5007)