import pandas as pd
import numpy as np
import schedule
import time
import joblib
from model_code.train_models import load_and_preprocess_data, create_sequences, train_traffic_model, train_fault_model, train_fault_classifier
from model_code.online_learning import OnlineLearner

def retrain_job():
    """Retrain models with feedback and new data."""
    feedback_file = 'agent_code/feedback_log.csv'
    data_file = 'model_code/historical_logs.csv'
    feedback_df = pd.read_csv(feedback_file)

    if len(feedback_df) < 100:
        print("Insufficient feedback for retraining.")
        return

    scaled_features, scaler, df = load_and_preprocess_data(data_file, MODEL_FEATURE_COLUMNS, EXTERNAL_FEATURES)
    joblib.dump(scaler, 'model_code/scaler.joblib')

    seq_length = 10
    target_feature = "dl_traffic"
    X_seq, y_seq = create_sequences(scaled_features, MODEL_FEATURE_COLUMNS.index(target_feature), seq_length)
    X_train_seq, _, y_train_seq, _ = train_test_split(X_seq, y_seq, test_size=0.2, random_state=42)
    traffic_model = train_traffic_model(X_train_seq, y_train_seq, seq_length, len(MODEL_FEATURE_COLUMNS + EXTERNAL_FEATURES))
    traffic_model.save('model_code/traffic_prediction_model.h5')

    X_train_fault, _ = train_test_split(scaled_features, test_size=0.2, random_state=42)
    fault_model = train_fault_model(X_train_fault)
    fault_model.save('model_code/fault_detection_model.h5')

    if 'fault_type' in feedback_df.columns:
        y_fault = feedback_df['fault_type'].fillna('none').values
        train_fault_classifier(X_train_fault, y_fault)
    else:
        pseudo_labels = generate_pseudo_labels(fault_model, X_train_fault)
        train_fault_classifier(X_train_fault, pseudo_labels)

    print("Retraining completed.")

schedule.every(7).days.do(retrain_job)

if __name__ == "__main__":
    traffic_learner = OnlineLearner('model_code/traffic_prediction_model.h5', len(MODEL_FEATURE_COLUMNS + EXTERNAL_FEATURES))
    fault_learner = OnlineLearner('model_code/fault_detection_model.h5', len(MODEL_FEATURE_COLUMNS + EXTERNAL_FEATURES))
    while True:
        schedule.run_pending()
        time.sleep(60)