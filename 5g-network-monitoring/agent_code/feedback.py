from flask import Flask, request
import pandas as pd
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
import base64
import logging

app = Flask(__name__)
logging.basicConfig(filename='agent_code/agent_log.txt', level=logging.INFO)

key = b'32_byte_key_for_AES256_1234567890'
iv = b'16_byte_iv_1234567890abc'

def encrypt_data(data):
    cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
    encryptor = cipher.encryptor()
    padded_data = data.encode() + b' ' * (16 - len(data) % 16)
    return base64.b64encode(encryptor.update(padded_data) + encryptor.finalize())

@app.route('/feedback', methods=['POST'])
def store_feedback():
    """Store feedback and RL rewards."""
    data = request.json
    prediction = data['prediction']
    action = data['action']
    timestamp = data['timestamp']
    feedback = data.get('feedback', '')
    reward = data.get('reward', 0.0)

    feedback_df = pd.DataFrame([{
        "prediction": str(prediction),
        "action": str(action),
        "timestamp": timestamp,
        "feedback": feedback,
        "reward": reward
    }])
    encrypted_data = encrypt_data(feedback_df.to_csv(index=False))
    with open('agent_code/feedback_log.csv', 'ab') as f:
        f.write(encrypted_data + b'\n')
    logging.info(f"Stored feedback: {feedback}, Reward: {reward}")
    return {"status": "success"}, 200

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5006)