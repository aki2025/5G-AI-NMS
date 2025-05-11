import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, MultiHeadAttention, LayerNormalization
from tensorflow.keras.models import Model
import numpy as np

def build_fusion_model(input_dim, output_dim=1):
    """Build transformer-based fusion model."""
    inputs = Input(shape=(input_dim,))
    x = Dense(64, activation='relu')(inputs)
    x = tf.expand_dims(x, axis=1)  # Add sequence dimension
    attention = MultiHeadAttention(num_heads=4, key_dim=16)(x, x)
    x = LayerNormalization()(x + attention)
    x = tf.squeeze(x, axis=1)
    x = Dense(32, activation='relu')(x)
    outputs = Dense(output_dim, activation='sigmoid')(x)  # Network health score
    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='mse')
    return model

def train_fusion_model(X_train, y_train, epochs=10):
    """Train and quantize fusion model."""
    model = build_fusion_model(input_dim=X_train.shape[1])
    model.fit(X_train, y_train, epochs=epochs, batch_size=32, validation_split=0.2, verbose=1)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    with open('model_code/fusion_model.tflite', 'wb') as f:
        f.write(tflite_model)
    return model

# Example usage (run separately for training)
if __name__ == "__main__":
    X_train = np.random.rand(1000, len(MODEL_FEATURE_COLUMNS + EXTERNAL_FEATURES))
    y_train = np.random.rand(1000, 1)
    train_fusion_model(X_train, y_train)