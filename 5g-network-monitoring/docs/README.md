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
