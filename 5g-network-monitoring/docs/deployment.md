# Deployment Guide

- **Edge**: Deploy microservices (`log_ingestion.py`, `orchestrator.py`, etc.) and `federated_client.py` on MEC servers (e.g., NVIDIA A100).
- **Cloud**: Run `federated_server.py`, `retrain_models.py`, and dashboard on AWS/Azure.
- **Kubernetes**: Use for orchestration, auto-scaling (HPA with CPU > 80%).
- **Security**: Enable TLS, monitor with Wazuh.
- **Latency**: Target ~50-100 ms per cycle for URLLC compatibility.