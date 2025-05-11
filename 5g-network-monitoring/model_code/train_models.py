import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
import shap
from tensorflow_model_optimization.sparsity import keras as sparsity
from model_code.multi_modal_fusion import train_fusion_model
from model_code.rl_agent import DQNAgent

# Define standard columns and external features
STANDARD_COLUMNS = [
    "log_time", "base_station_id", "concurrent_users", "ul_traffic", "dl_traffic",
    "ul_throughput", "dl_throughput", "latency_ms", "packet_loss_rate", "rsrp_dbm",
    "rsrq_db", "sinr_db", "cpu_usage", "mem_usage", "urllc_latency", "isac_signal"
]
MODEL_FEATURE_COLUMNS = STANDARD_COLUMNS[2:]
EXTERNAL_FEATURES = ["temperature", "precipitation", "congestion_score", "event_density", "urban_density", "social_trend_score"]

def load_and_preprocess_data(file_path, model_feature_columns, external_features, mapping=None):
    """Load and preprocess historical data."""
    df = pd.read_csv(file_path)
    if mapping:
        df = df.rename(columns={v: k for k, v in mapping.items()})
    for col in STANDARD_COLUMNS + external_features:
        if col not in df.columns:
            df[col] = np.nan
    df['log_time'] = pd.to_datetime(df['log_time'])
    df = df.sort_values('log_time')
    features = df[model_feature_columns + external_features].astype(float)
    features = features.fillna(method='ffill').fillna(method='bfill')
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    return scaled_features, scaler, df

def create_sequences(scaled_features, target_index, seq_length):
    """Create sequences for LSTM training."""
    X, y = [], []
    for i in range(len(scaled_features) - seq_length):
        X.append(scaled_features[i:i + seq_length])
        y.append(scaled_features[i + seq_length, target_index])
    return np.array(X), np.array(y)

def train_traffic_model(X_train, y_train, seq_length, feature_dim):
    """Train, prune, and quantize LSTM model."""
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(64, activation='relu', input_shape=(seq_length, feature_dim), return_sequences=True),
        tf.keras.layers.LSTM(32, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    pruning_params = {'pruning_schedule': sparsity.PolynomialDecay(0, 0.5, 0, 1000)}
    model = sparsity.prune_low_magnitude(model, **pruning_params)
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2, verbose=1)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    with open('traffic_prediction_model.tflite', 'wb') as f:
        f.write(tflite_model)
    return model

def train_fault_model(X_train):
    """Train, prune, and quantize Autoencoder model."""
    input_dim = X_train.shape[1]
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(input_dim,)),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(input_dim, activation='linear')
    ])
    pruning_params = {'pruning_schedule': sparsity.PolynomialDecay(0, 0.5, 0, 1000)}
    model = sparsity.prune_low_magnitude(model, **pruning_params)
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, X_train, epochs=20, batch_size=32, validation_split=0.2, verbose=1)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    with open('fault_detection_model.tflite', 'wb') as f:
        f.write(tflite_model)
    return model

def generate_pseudo_labels(autoencoder, data, threshold=0.1):
    """Generate pseudo-labels using Autoencoder."""
    recon = autoencoder.predict(data)
    mse = np.mean((data - recon) ** 2, axis=1)
    return ['anomaly' if e > threshold else 'normal' for e in mse]

def train_fault_classifier(X_train, y_train):
    """Train Random Forest with SHAP explainer."""
    classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    classifier.fit(X_train, y_train)
    explainer = shap.TreeExplainer(classifier)
    joblib.dump(classifier, 'fault_classifier_model.pkl')
    joblib.dump(explainer, 'fault_explainer.pkl')
    return classifier, explainer

def main():
    """Train and save all models."""
    data_file = 'historical_logs.csv'
    seq_length = 10
    target_feature = "dl_traffic"

    scaled_features, scaler, df = load_and_preprocess_data(data_file, MODEL_FEATURE_COLUMNS, EXTERNAL_FEATURES)
    joblib.dump(scaler, 'scaler.joblib')

    X_seq, y_seq = create_sequences(scaled_features, MODEL_FEATURE_COLUMNS.index(target_feature), seq_length)
    X_train_seq, X_val_seq, y_train_seq, y_val_seq = train_test_split(X_seq, y_seq, test_size=0.2, random_state=42)
    traffic_model = train_traffic_model(X_train_seq, y_train_seq, seq_length, len(MODEL_FEATURE_COLUMNS + EXTERNAL_FEATURES))
    traffic_model.save('traffic_prediction_model.h5')

    X_train_fault, X_val_fault = train_test_split(scaled_features, test_size=0.2, random_state=42)
    fault_model = train_fault_model(X_train_fault)
    fault_model.save('fault_detection_model.h5')

    if 'fault_type' not in df.columns:
        pseudo_labels = generate_pseudo_labels(fault_model, X_train_fault)
        train_fault_classifier(X_train_fault, pseudo_labels)
    else:
        y_fault = df['fault_type'].fillna('none').values
        X_train_fault, X_val_fault, y_train_fault, y_val_fault = train_test_split(scaled_features, y_fault, test_size=0.2, random_state=42)
        train_fault_classifier(X_train_fault, y_train_fault)

    train_fusion_model(X_train_fault, np.random.rand(len(X_train_fault), 1))  # Placeholder for fusion training
    rl_agent = DQNAgent(state_dim=len(MODEL_FEATURE_COLUMNS + EXTERNAL_FEATURES) + 3, action_dim=5)
    rl_agent.save()

if __name__ == "__main__":
    main()