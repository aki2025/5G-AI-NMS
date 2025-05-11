import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model
import joblib

class DQNAgent:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.model = self.build_model()
        self.target_model = self.build_model()
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.memory = []
        self.gamma = 0.95
        self.batch_size = 32

    def build_model(self):
        """Build DQN model."""
        inputs = Input(shape=(self.state_dim,))
        x = Dense(64, activation='relu')(inputs)
        x = Dense(32, activation='relu')(x)
        outputs = Dense(self.action_dim, activation='linear')(x)
        model = Model(inputs, outputs)
        model.compile(optimizer='adam', loss='mse')
        return model

    def act(self, state):
        """Select action using epsilon-greedy policy."""
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_dim)
        state = np.array(state).reshape(1, -1)
        q_values = self.model.predict(state, verbose=0)
        return np.argmax(q_values[0])

    def train(self, state, action, reward, next_state, done):
        """Train DQN with experience replay."""
        self.memory.append((state, action, reward, next_state, done))
        if len(self.memory) < self.batch_size:
            return
        batch = np.random.choice(len(self.memory), self.batch_size, replace=False)
        states = np.array([self.memory[i][0] for i in batch])
        actions = np.array([self.memory[i][1] for i in batch])
        rewards = np.array([self.memory[i][2] for i in batch])
        next_states = np.array([self.memory[i][3] for i in batch])
        dones = np.array([self.memory[i][4] for i in batch])

        targets = self.model.predict(states, verbose=0)
        target_next = self.target_model.predict(next_states, verbose=0)
        for i in range(self.batch_size):
            if dones[i]:
                targets[i][actions[i]] = rewards[i]
            else:
                targets[i][actions[i]] = rewards[i] + self.gamma * np.max(target_next[i])
        self.model.fit(states, targets, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_target_model(self):
        """Update target model weights."""
        self.target_model.set_weights(self.model.get_weights())

    def save(self, path='model_code/rl_policy.pkl'):
        """Save DQN model."""
        joblib.dump(self.model, path)

# Example training (run separately)
if __name__ == "__main__":
    agent = DQNAgent(state_dim=len(MODEL_FEATURE_COLUMNS + EXTERNAL_FEATURES) + 3, action_dim=5)
    agent.save()