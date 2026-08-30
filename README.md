# MLOps Assignment 3 — PyTorch Image Classification on Kubernetes

## 1. Project Overview

This project demonstrates an end-to-end MLOps workflow for CIFAR-10 image classification using **PyTorch, Docker, and Kubernetes**.

The workflow covers:

Repository and Git workflow
PyTorch model development
CIFAR-10 dataset preparation
Containerized model training
Kubernetes Job-based training
Persistent storage for datasets and model checkpoints
Kubernetes model serving
FastAPI prediction API
Health and readiness checks
Kubernetes Service
Horizontal Pod Autoscaling (HPA)
End-to-end validation

The classification model is a modified **ResNet18** trained on the CIFAR-10 dataset.

## 2. Technology Stack

| Component            | Technology                       |
| -------------------- | -------------------------------- |
| Programming Language | Python 3.11                      |
| Deep Learning        | PyTorch                          |
| Dataset              | CIFAR-10                         |
| Model                | ResNet18                         |
| API                  | FastAPI                          |
| ASGI Server          | Uvicorn                          |
| Containerization     | Docker                           |
| Orchestration        | Kubernetes                       |
| Local Kubernetes     | Docker Desktop Kubernetes        |
| Storage              | Kubernetes PersistentVolumeClaim |
| Configuration        | Kubernetes ConfigMap             |
| Autoscaling          | Kubernetes HPA                   |
| Version Control      | Git / GitHub                     |


# 3. Architecture

The project separates model training from model serving.

```text
                         ┌──────────────────────┐
                         │      Developer       │
                         │      Git / GitHub    │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │   Docker Images      │
                         │                      │
                         │  Trainer             │
                         │  Serving             │
                         └──────────┬───────────┘
                                    │
                  ┌─────────────────┴─────────────────┐
                  │                                   │
                  ▼                                   ▼
       ┌──────────────────────┐             ┌──────────────────────┐
       │ Kubernetes Training  │             │ Kubernetes Serving   │
       │ Job                  │             │ Deployment           │
       │                      │             │                      │
       │ PyTorch ResNet18     │             │ 2 Replicas           │
       │ CIFAR-10 Training    │             │ FastAPI + Uvicorn    │
       └──────────┬───────────┘             └──────────┬───────────┘
                  │                                    │
                  │                                    │
                  ▼                                    ▼
       ┌──────────────────────┐             ┌──────────────────────┐
       │ Training Data PVC    │             │ ClusterIP Service    │
       │                      │             │ Port 80              │
       │ CIFAR-10 dataset     │             │ → Container :8080    │
       └──────────────────────┘             └──────────┬───────────┘
                                                       │
                                                       ▼
                                            ┌──────────────────────┐
                                            │ FastAPI              │
                                            │                      │
                                            │ GET  /health         │
                                            │ POST /predict        │
                                            └──────────┬───────────┘
                                                       │
                                                       ▼
                                            ┌──────────────────────┐
                                            │ ResNet18 Model       │
                                            │                      │
                                            │ classifier_v1.pt     │
                                            └──────────────────────┘


       ┌──────────────────────────────────────────────────────────┐
       │              Shared Checkpoint PVC                       │
       │                                                          │
       │              classifier_v1.pt                            │
       │                                                          │
       │   Training Job ──► writes checkpoint                      │
       │   Serving Pods ──► read checkpoint (read-only)            │
       └──────────────────────────────────────────────────────────┘
```

### Serving flow

```text
Client
  │
  │ POST /predict
  │ image
  ▼
ClusterIP Service :80
  │
  ├──────────────► Serving Pod 1 :8080
  │
  └──────────────► Serving Pod 2 :8080
                         │
                         ▼
                    ResNet18
                         │
                         ▼
                  CIFAR-10 class
```

---

# 4. Repository Structure

```text
mlops-pytorch-pipeline/
│
├── configs/
│   └── training_config.yaml
│
├── docker/
│   ├── Dockerfile.train
│   └── Dockerfile.serve
│
├── k8s/
│   ├── namespace.yaml
│   ├── configmap.yaml
│   ├── training-pvc.yaml
│   ├── training-job.yaml
│   ├── serving-deployment.yaml
│   ├── serving-service.yaml
│   └── hpa.yaml
│
├── requirements/
│   ├── train.txt
│   └── serve.txt
│
├── src/
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   └── serve.py
│
├── tests/
│
├── README.md
└── .gitignore
```

---

# 5. Prerequisites

Install the following:

* Python 3.11
* Git
* Docker Desktop
* Kubernetes enabled in Docker Desktop
* `kubectl`

Verify the installations:

```bash
python --version
git --version
docker --version
kubectl version --client
kubectl get nodes
```

The Kubernetes cluster should report the Docker Desktop node as `Ready`.

---

# 6. Local Python Environment

Create and activate a virtual environment.

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install training dependencies:

```powershell
pip install -r requirements/train.txt
```

Install serving dependencies:

```powershell
pip install -r requirements/serve.txt
```

---

# 7. Build Docker Images

Build the training image:

```powershell
docker build -f docker/Dockerfile.train -t cifar10-trainer:latest .
```

Build the serving image:

```powershell
docker build -f docker/Dockerfile.serve -t cifar10-serving:latest .
```

Verify:

```powershell
docker images | Select-String "cifar10"
```

For Docker Desktop Kubernetes, the locally built images can be used with:

```yaml
imagePullPolicy: IfNotPresent
```

---

# 8. Kubernetes Deployment

Create the namespace:

```powershell
kubectl apply -f k8s/namespace.yaml
```

Apply configuration:

```powershell
kubectl apply -f k8s/configmap.yaml
```

Create persistent storage:

```powershell
kubectl apply -f k8s/training-pvc.yaml
```

Verify the PVCs:

```powershell
kubectl get pvc -n ml-training
```

---

# 9. Run Model Training

Create the Kubernetes training Job:

```powershell
kubectl apply -f k8s/training-job.yaml
```

Check the Job:

```powershell
kubectl get jobs -n ml-training
```

Check the training pod:

```powershell
kubectl get pods -n ml-training
```

View training logs:

```powershell
kubectl logs -n ml-training <training-pod-name>
```

The training process produces the model checkpoint:

```text
/app/checkpoints/classifier_v1.pt
```

Verify the checkpoint:

```powershell
kubectl exec -n ml-training <training-pod-name> -- ls -lh /app/checkpoints
```

---

# 10. Deploy Model Serving

After the model checkpoint is available, deploy the serving layer:

```powershell
kubectl apply -f k8s/serving-deployment.yaml
```

Create the Service:

```powershell
kubectl apply -f k8s/serving-service.yaml
```

Deploy the HPA:

```powershell
kubectl apply -f k8s/hpa.yaml
```

Verify:

```powershell
kubectl get deployment -n ml-training
kubectl get pods -n ml-training
kubectl get svc -n ml-training
kubectl get hpa -n ml-training
```

Expected serving deployment:

```text
cifar10-serving   2/2   2   2
```

---

# 11. Serving Configuration

The serving Deployment provides:

* 2 replicas
* RollingUpdate strategy
* `maxSurge: 1`
* `maxUnavailable: 0`
* CPU request: `500m`
* Memory request: `1Gi`
* CPU limit: `1`
* Memory limit: `2Gi`
* Liveness probe: `GET /health`
* Liveness interval: 10 seconds
* Liveness failure threshold: 3
* Readiness probe: `GET /health`
* Readiness interval: 5 seconds
* Readiness initial delay: 15 seconds
* Read-only checkpoint volume

The Kubernetes Service exposes:

```text
Service port: 80
Container port: 8080
Service type: ClusterIP
```

---

# 12. Health Check

For local testing, port-forward the Service:

```powershell
kubectl port-forward -n ml-training service/cifar10-serving 8090:80
```

Then:

```powershell
curl.exe http://localhost:8090/health
```

Example response:

```json
{
  "status": "healthy",
  "model": "resnet18",
  "device": "cpu"
}
```

---

# 13. Prediction API

The prediction endpoint accepts an image file:

```text
POST /predict
```

Example:

```powershell
curl.exe -X POST http://localhost:8090/predict -F "image=@test_image.png"
```

Example successful response:

```json
{
  "predicted_class": "cat",
  "predicted_index": 3,
  "probabilities": {
    "airplane": 4e-06,
    "automobile": 5e-06,
    "bird": 4.1e-05,
    "cat": 0.999673,
    "deer": 0.0,
    "dog": 0.000272,
    "frog": 1e-06,
    "horse": 0.0,
    "ship": 3e-06,
    "truck": 1e-06
  }
}
```

This demonstrates the complete path:

```text
CIFAR-10 image
      ↓
Kubernetes ClusterIP Service
      ↓
FastAPI serving pod
      ↓
ResNet18
      ↓
Predicted CIFAR-10 class
```

---

# 14. End-to-End Validation

Verify all Kubernetes resources:

```powershell
kubectl get all -n ml-training
```

Verify persistent volumes:

```powershell
kubectl get pvc -n ml-training
```

Verify configuration:

```powershell
kubectl get configmap -n ml-training
```

Verify autoscaling:

```powershell
kubectl get hpa -n ml-training
```

Describe the serving Deployment:

```powershell
kubectl describe deployment cifar10-serving -n ml-training
```

Verify the checkpoint from a serving pod:

```powershell
kubectl exec -n ml-training <serving-pod-name> -- ls -lh /app/checkpoints
```

The final validation should demonstrate:

* Training Job created
* Model checkpoint generated
* Checkpoint persisted using PVC
* Two serving replicas running
* Health checks passing
* ClusterIP Service available
* HPA configured
* `/health` returning healthy
* `/predict` returning a CIFAR-10 prediction

---

# 15. Kubernetes Architecture Summary

```text
                 GitHub
                    │
                    ▼
             Docker Images
              │          │
              ▼          ▼
        Training Job   Serving Deployment
              │          │
              │       ┌──┴──┐
              │       ▼     ▼
              │     Pod 1  Pod 2
              │       │     │
              └──► Checkpoint PVC
                         │
                         ▼
                    classifier_v1.pt

Client
  │
  ▼
ClusterIP Service :80
  │
  ▼
Serving Pods :8080
  │
  ▼
FastAPI /predict
  │
  ▼
ResNet18
  │
  ▼
CIFAR-10 Prediction
```

---

# 16. Project Outcome

The project demonstrates a reproducible Kubernetes-based machine learning workflow in which:

1. CIFAR-10 data is provided to a Kubernetes training Job.
2. ResNet18 is trained using PyTorch.
3. The trained checkpoint is persisted to a Kubernetes PVC.
4. The serving Deployment mounts the checkpoint read-only.
5. Two FastAPI serving replicas load the model.
6. Kubernetes health probes verify application availability.
7. A ClusterIP Service provides internal access to the prediction API.
8. HPA provides CPU-based autoscaling configuration.
9. An end-to-end image request successfully produces a CIFAR-10 prediction.
