import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model

class OnlineLearner:
    def __init__(self, model_path, feature_dim):
        self.model = load_model(model_path)
        self.feature_dim = feature_dim
        self.batch_size = 32
        self.buffer = []

    def update(self, features, target):
        """Update model incrementally."""
        self.buffer.append((features, target))
        if len(self.buffer) >= self.batch_size:
            X = np.array([b[0] for b in self.buffer])
            y = np.array([b[1] for b in self.buffer])
            self.model.fit(X, y, epochs=1, batch_size=self.batch_size, verbose=0)
            self.buffer = []

    def save(self, path):
        """Save updated model."""
        self.model.save(path)