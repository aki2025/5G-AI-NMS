from flask import Flask, render_template, request, redirect
from flask_login import LoginManager, UserMixin, login_required, login_user
import pandas as pd
from requests_oauthlib import OAuth2Session

app = Flask(__name__)
app.secret_key = 'your_secret_key'
login_manager = LoginManager(app)

class User(UserMixin):
    def __init__(self, id, role):
        self.id = id
        self.role = role

users = {'admin': User('admin', 'admin'), 'operator': User('operator', 'operator')}

@login_manager.user_loader
def load_user(user_id):
    return users.get(user_id)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        user_id = request.form['username']
        if user_id in users:
            login_user(users[user_id])
            return redirect('/')
    return render_template('login.html')

@app.route('/')
@login_required
def home():
    feedback_df = pd.read_csv('agent_code/feedback_log.csv')
    latest_pred = feedback_df.iloc[-1]
    return render_template('index.html', prediction=eval(latest_pred['prediction']), action=eval(latest_pred['action']))

@app.route('/logs')
@login_required
def logs():
    with open('agent_code/agent_log.txt', 'r') as f:
        logs = f.readlines()[-50:]
    return render_template('logs.html', logs=logs)

@app.route('/feedback', methods=['GET', 'POST'])
@login_required
def feedback():
    if request.method == 'POST':
        feedback = request.form['feedback']
        requests.post('http://localhost:5006/feedback', json={"feedback": feedback, "reward": 1.0})
        return "Feedback submitted."
    return render_template('feedback.html')

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5001)