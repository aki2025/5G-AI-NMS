# 5G Network Monitoring System

**AI-based system for real-time 5G network monitoring with enhanced intelligence.**

## Enhanced Features

- **Multi-Modal Fusion**: Transformer-based fusion of logs, social media, and GIS data.
- **Reinforcement Learning**: DQN for optimized action selection.
- **Online Learning**: Real-time model updates for adaptability.
- **Automated Execution**: Integration with network orchestrators (e.g., ONAP).
- **Distributed Architecture**: Microservices for scalability.
- **Security**: RBAC, OAuth, AES-256 encryption.

## Setup Instructions

1. Install dependencies: `pip install -r requirements.txt`.
2. Start Kafka: `docker run -p 9092:9092 apache/kafka`.
3. Train models: `python model_code/train_models.py`.
4. Run microservices: `docker-compose up`.
5. Run federated server: `python federated_learning/federated_server.py`.
6. Launch dashboard: `python dashboard/app.py`.

## Code Folder Structure 5g-network-monitoring

	## model_code/
	 ├── train_models.py              # Trains base models with quantization and SHAP
         ├── retrain_models.py            # Automated retraining with feedback
         ├── online_learning.py           # Online model updates for real-time adaptation
         ├── multi_modal_fusion.py        # Transformer-based fusion of multi-modal data
         ├── rl_agent.py                  # RL agent for action optimization
         ├── historical_logs.csv          # Sample dataset (not included, user-provided)
         ├── scaler.joblib                # Saved scaler
         ├── traffic_prediction_model.tflite  # Quantized LSTM model
         ├── fault_detection_model.tflite     # Quantized Autoencoder model
         ├── fault_classifier_model.pkl       # Random Forest model
         ├── fault_explainer.pkl          # SHAP explainer
         ├── fusion_model.tflite          # Quantized fusion model
         ├── rl_policy.pkl                # RL policy network
         └── 3gpp_parser.py               # Parses 3GPP documents
      
	## agent_code/
          ├── log_ingestion.py             # Fetches and streams logs to Kafka
          ├── preprocessing.py             # Dynamic mapping and scaling
          ├── prediction.py                # Model inference with fusion
          ├── action.py                    # Action suggestions with RL
          ├── feedback.py                  # Feedback processing
          ├── orchestrator.py              # Automated action execution
          ├── agent_log.txt                # Log file
          ├── config.json                  # Configuration
          │── feedback_log.csv             # Feedback storage


	## dashborad
           ├── app.py                       # Flask dashboard with RBAC
           ├── templates/                   # HTML templates
           │   ├── index.html
           │   ├── logs.html
           │   ├── feedback.html
           │   └── login.html
           └── static/                      # Static files

	## federated_learning/
			 ├── federated_client.py          # Edge training with differential privacy
			 └── federated_server.py          # Weight aggregation

	## docs/
             ├── README.md                    # Project overview
             ├── deployment.md                # Deployment guide
             └── use_cases.md                 # Use cases


    ├── requirements.txt                 # Dependencies
    ├── docker-compose.yml               # Docker configuration
    └── .gitignore                       # Files to ignore


## Notes
## Prerequisites:
    1.	Install Docker and Kafka (docker run -p 9092:9092 apache/kafka).  
    2.	Provide historical_logs.csv or generate dummy data:

    python

    -	import pandas as pd
    -	import numpy as np
    -	data = {col: np.random.rand(1000) for col in STANDARD_COLUMNS[2:]}
    -	data['log_time'] = pd.date_range('2023-01-01', periods=1000, freq='H')
    -	data['base_station_id'] = np.random.randint(1, 10, 1000)
    -	pd.DataFrame(data).to_csv('model_code/historical_logs.csv', index=False)

    3.	Replace API keys in config.json and orchestrator URL in orchestrator.py.  
    4.	Securely manage AES key and IV in feedback.py (e.g., AWS KMS).

## Running the system
      Train models: python model_code/train_models.py.  
      Start Kafka and microservices: docker-compose up.  
      Run federated server: python federated_learning/federated_server.py.  
      Launch dashboard: python dashboard/app.py.  
      Simulate edge training: python federated_learning/federated_client.py.

## Latency Performance:

      Log Ingestion: ~10-50 ms (cached external data).  
      Preprocessing + Fusion: ~10-15 ms (transformer inference).  
      Prediction + Online Learning: ~20-30 ms (LSTM + Autoencoder + updates).  
      Action + RL: ~10-15 ms (DQN inference).  
      Orchestration: ~50-100 ms (ONAP API call).  
      Total: ~60-110 ms per cycle, suitable for URLLC and general monitoring.




