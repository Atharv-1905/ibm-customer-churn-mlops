# IBM Customer Churn MLOps

An end-to-end MLOps project for predicting IBM Telco customer churn using a Scikit-learn machine-learning pipeline and deploying the model as a FastAPI service with Docker and Kubernetes.

## Project Overview

This project demonstrates the complete ML lifecycle:

```text
Telco Customer Churn Dataset
        ↓
Feature Selection & Preprocessing
        ↓
Logistic Regression Model
        ↓
MLflow Experiment Tracking
        ↓
FastAPI REST API
        ↓
Docker Container
        ↓
GitHub Actions CI
        ↓
Kubernetes / Minikube
        ↓
Live Churn Prediction API
```

## Dataset

Dataset file:

```text
data/Telco_customer_churn.xlsx
```

Selected input features:

- `Monthly Charges`
- `Contract`
- `Paperless Billing`

Target:

- `Churn Value`

The model uses the selected numerical feature directly and one-hot encodes the categorical features.

## Machine Learning Model

The project uses a Scikit-learn `Pipeline` containing:

1. `ColumnTransformer`
2. `OneHotEncoder(handle_unknown="ignore")`
3. `LogisticRegression(max_iter=1000)`

Training configuration:

- Test size: `0.2`
- Random state: `42`
- Stratified train/test split

### Model Performance

| Metric | Score |
|---|---:|
| Accuracy | 0.7402 |
| Precision | 0.5159 |
| Recall | 0.3476 |
| F1 Score | 0.4153 |

The trained pipeline is saved as:

```text
model/model.pkl
```

## Project Structure

```text
ibm-customer-churn-mlops/
├── api/
│   ├── __init__.py
│   └── app.py
├── data/
│   └── Telco_customer_churn.xlsx
├── k8s/
│   ├── deployment.yaml
│   └── service.yaml
├── model/
│   └── model.pkl
├── src/
│   ├── predict.py
│   └── train.py
├── tests/
│   ├── __init__.py
│   └── test_api.py
├── .github/
│   └── workflows/
│       └── ci.yml
├── Dockerfile
├── .gitignore
└── requirements.txt
```

## 1. Environment Setup

Create and activate the virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## 2. Train the Model

From the project root:

```bash
python src/train.py
```

This:

- Loads the Excel dataset.
- Selects the three model features.
- Splits the data into training and testing sets.
- Trains the Logistic Regression pipeline.
- Calculates evaluation metrics.
- Logs parameters, metrics and the model to MLflow.
- Saves the final pipeline to `model/model.pkl`.

## 3. MLflow

Start MLflow from the project directory:

```bash
source .venv/bin/activate
mlflow ui
```

Open:

```text
http://127.0.0.1:5000
```

Experiment:

```text
IBM-Customer-Churn
```

MLflow records:

- Model parameters
- Accuracy
- Precision
- Recall
- F1 score
- Model artifact

## 4. Run FastAPI

Start the API:

```bash
uvicorn api.app:app --host 0.0.0.0 --port 8000
```

Health endpoint:

```bash
curl http://127.0.0.1:8000/health
```

Expected:

```json
{"status":"healthy"}
```

### Prediction Endpoint

Request:

```bash
curl -X POST http://127.0.0.1:8000/predict \
-H "Content-Type: application/json" \
-d '{"Monthly Charges":85.50,"Contract":"Month-to-month","Paperless Billing":"Yes"}'
```

Example response:

```json
{"churn_prediction":"Yes","churn_value":1}
```

Interactive Swagger documentation:

```text
http://127.0.0.1:8000/docs
```

## 5. Docker

Build the Docker image:

```bash
docker build -t churn-api:1.0 .
```

Run the container:

```bash
docker run -d --name churn-api -p 8000:8000 churn-api:1.0
```

Check the container:

```bash
docker ps
```

Test the API:

```bash
curl http://127.0.0.1:8000/health
```

View logs:

```bash
docker logs churn-api
```

Swagger:

```text
http://127.0.0.1:8000/docs
```

## 6. Testing and CI

Run tests locally:

```bash
pytest
```

The GitHub Actions workflow is located at:

```text
.github/workflows/ci.yml
```

The CI pipeline:

1. Checks out the repository.
2. Sets up Python.
3. Installs project dependencies.
4. Runs `pytest`.

A successful GitHub Actions run verifies that the API tests pass automatically on repository changes.

## 7. Kubernetes / Minikube

Check Minikube:

```bash
minikube status
```

Verify the image:

```bash
minikube image ls | grep churn-api
```

Apply the Deployment:

```bash
kubectl apply -f k8s/deployment.yaml
```

Apply the Service:

```bash
kubectl apply -f k8s/service.yaml
```

Check the Pod:

```bash
kubectl get pods
```

Check the Deployment:

```bash
kubectl get deployments
```

Check the Service:

```bash
kubectl get services
```

Get the service URL:

```bash
minikube service churn-api --url
```

Health check:

```bash
curl http://192.168.49.2:30080/health
```

Prediction:

```bash
curl -X POST http://192.168.49.2:30080/predict \
-H "Content-Type: application/json" \
-d '{"Monthly Charges":85.50,"Contract":"Month-to-month","Paperless Billing":"Yes"}'
```

Swagger through Kubernetes:

```text
http://192.168.49.2:30080/docs
```

> Note: The Minikube IP/URL can change after restarting Minikube. Use the URL returned by `minikube service churn-api --url`.

## Kubernetes Architecture

```text
                    Kubernetes
                        │
                 ┌──────▼──────┐
                 │  Deployment │
                 │  churn-api  │
                 └──────┬──────┘
                        │
                 ┌──────▼──────┐
                 │     Pod     │
                 │  FastAPI +  │
                 │  ML Model   │
                 │   :8000     │
                 └──────┬──────┘
                        │
                 ┌──────▼──────┐
                 │   NodePort  │
                 │    :30080   │
                 └─────────────┘
```

## MLOps Components

| Component | Purpose |
|---|---|
| Python / Scikit-learn | Model development and training |
| Pandas | Dataset loading and preparation |
| MLflow | Experiment tracking and model logging |
| FastAPI | REST API serving the model |
| Docker | Containerization |
| Pytest | Automated testing |
| GitHub Actions | CI automation |
| Kubernetes | Container orchestration |
| Minikube | Local Kubernetes environment |
| kubectl | Kubernetes management |

## Quick Demo

For a short project demonstration:

### MLflow

```bash
source .venv/bin/activate
mlflow ui
```

Open `http://127.0.0.1:5000` and show `IBM-Customer-Churn` and its metrics.

### Docker

```bash
docker ps
curl http://127.0.0.1:8000/health
```

Then test `/predict`.

### Kubernetes

```bash
minikube status
kubectl get pods
kubectl get deployments
kubectl get services
```

Then test:

```bash
curl http://192.168.49.2:30080/health
```

and the `/predict` endpoint.

## Git Workflow

Check repository status:

```bash
git status
```

Add changes:

```bash
git add .
```

Commit:

```bash
git commit -m "Update project documentation"
```

Push to GitHub:

```bash
git push origin master
```

## Author

**Atharv Bhosale**

## Project Goal

The goal of this project is to demonstrate an end-to-end MLOps workflow, from customer churn model development and experiment tracking to API serving, containerization, continuous integration, monitoring, and Kubernetes deployment.
