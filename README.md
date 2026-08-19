# U.S. Flight Delay Prediction & Ops Dashboard

## Project Overview

This project implements a production-style MLOps pipeline for predicting U.S. flight arrival delays using the U.S. DOT/BTS On-Time Performance dataset.

The system predicts continuous arrival delay in minutes and also derives whether a flight is expected to arrive more than 15 minutes late.

The project includes:

- Model experimentation and tracking with Weights & Biases
- Model versioning and registry
- FastAPI prediction service
- PostgreSQL prediction logging
- Streamlit user application
- Separate Streamlit monitoring dashboard
- Dockerized application components
- AWS EC2 deployment
- AWS RDS PostgreSQL
- Automated testing and CI with GitHub Actions

---

## Problem Statement

Predict arrival delays and provide route-level and operational visibility for travelers and operations users.

The primary machine learning target is:

`ARR_DELAY`

The model predicts arrival delay in minutes.

A secondary classification is derived from the regression output:

- 15 minutes or less
- More than 15 minutes late

---

## Dataset

Source:

U.S. Department of Transportation / Bureau of Transportation Statistics On-Time Performance dataset.

The model uses flight information available before departure.

Example features include:

- Day of month
- Day of week
- Operating airline
- Tail number
- Flight number
- Origin airport
- Destination airport
- Scheduled departure time
- Scheduled arrival time
- Scheduled elapsed time

The model target is:

`ARR_DELAY`

The data was split chronologically:

- Training: January through May 2026
- Testing: June 2026

The raw dataset is not included in the GitHub repository because of file size limitations.

---

## Machine Learning

Multiple experiments were evaluated during development.

Models included:

- Linear Regression ( 2 models, the second Linear Regression feature set removed TAIL_NUM and OP_CARRIER_FL_NUM)


Linear Regression was selected as the production baseline.

The model is implemented as a Scikit-learn pipeline containing preprocessing and regression.

Preprocessing includes:

- One-hot encoding for categorical variables
- Standard scaling for numeric variables

---

## Experiment Tracking

Weights & Biases is used for experiment tracking.

Tracked information includes:

- Model type
- Hyperparameters
- Training period
- Test period
- Feature set
- Data version
- MAE
- RMSE
- R²
- Precision
- Recall
- F1 score
- Git commit metadata

Trained models are stored as W&B artifacts.

The selected model is promoted through the W&B Model Registry and loaded by the production API.

Screenshots are stored in the `docs/` folder.

---

## System Architecture

The production system contains four major components.

### 1. FastAPI Backend

The FastAPI service:

- Loads the production model from the W&B Model Registry
- Accepts prediction requests
- Generates arrival delay predictions
- Calculates whether the predicted delay exceeds 15 minutes
- Measures prediction latency
- Logs predictions to PostgreSQL

Endpoints:

`GET /health`

Checks API and model availability.

`POST /predict`

Accepts flight information and returns a prediction.

Example response:

```json
{
  "prediction_id": 1,
  "predicted_arrival_delay_minutes": 8.53,
  "delay_over_15_minutes": false,
  "prediction_latency_ms": 6.5
}
```

---

### 2. PostgreSQL Database

AWS RDS PostgreSQL is used as the persistent database.

Each prediction stores information including:

- Prediction ID
- Timestamp
- Flight information
- Predicted delay
- More-than-15-minute indicator
- Prediction latency
- Actual delay
- Feedback correctness
- Feedback timestamp

The database allows the frontend, API, and monitoring system to exchange persistent information without relying on local JSON files.

---

### 3. User Frontend

A Streamlit user interface allows users to enter flight information and request a prediction.

The application sends the request to the FastAPI backend and displays:

- Predicted arrival delay
- Whether the predicted delay exceeds 15 minutes
- Prediction ID

The frontend runs on a separate AWS EC2 instance.

Screenshots are stored in the `docs/` folder.

---

### 4. Monitoring Dashboard

A separate Streamlit monitoring application provides operational visibility into the deployed model.

The dashboard displays:

- Total predictions
- Average prediction latency
- Prediction latency over time
- Predicted class distribution
- Target drift over time
- Recent prediction logs
- User feedback
- Live classification accuracy from submitted feedback

The monitoring application connects directly to PostgreSQL and runs on a separate AWS EC2 instance.

Screenshots are stored in the `docs/` folder.

---

## Target Drift Monitoring

Target drift is monitored using the proportion of predictions classified as more than 15 minutes late.

The dashboard tracks how this percentage changes over time.

This provides a simple way to identify changes in the distribution of model predictions after deployment.

---

## Feedback and Live Accuracy

The monitoring dashboard includes a feedback mechanism.

After the actual flight result is known, a user can enter:

- Prediction ID
- Actual arrival delay

The system compares the actual delay with the previously predicted classification.

The database stores whether the classification was correct.

The monitoring dashboard then calculates live accuracy using available feedback records.

---

## AWS Deployment

The system is deployed entirely on AWS.

### EC2 Instance 1
FastAPI backend

Port:

`8000`

### EC2 Instance 2
User-facing Streamlit application

Port:

`8501`

### EC2 Instance 3
Monitoring Streamlit dashboard

Port:

`8501`

### AWS RDS
PostgreSQL database used for persistent prediction and feedback storage.

Screenshots of the AWS deployment are stored in the `docs/` folder.

---

## Docker

Each application component is containerized using Docker.

Docker images include:

- FastAPI backend
- Streamlit frontend
- Streamlit monitoring dashboard

Images are built for the Linux AMD64 platform for compatibility with the AWS EC2 environment.

Example:

```bash
docker buildx build \
  --platform linux/amd64 \
  -t <dockerhub-user>/<image-name>:latest \
  --push \
  .
```

Containers are then pulled and executed on their respective EC2 instances.

---

## Testing

Pytest is used for automated testing.

Tests include:

### Health Endpoint Test

Confirms that:

`GET /health`

returns a successful response.

### Prediction Endpoint Test

Confirms that:

`POST /predict`

accepts valid input and returns the expected prediction response structure.

External dependencies such as the database and prediction model are mocked during testing.

Run tests locally with:

```bash
pytest
```

---

## Linting

Ruff is used for code quality and linting.

Run locally with:

```bash
ruff check .
```

---

## Continuous Integration

GitHub Actions provides automated Continuous Integration.

For each pull request into `main`, GitHub Actions:

1. Checks out the repository
2. Configures Python
3. Installs dependencies
4. Runs Ruff
5. Runs Pytest

If linting or tests fail, the CI check fails.

The `main` branch is protected with a ruleset requiring the CI status check to pass before code can be merged.

Screenshots of the successful GitHub Actions workflow are stored in the `docs/` folder.

---

## Project Structure

```text
MLOpsSummer26-Project/
│
├── .github/
│   └── workflows/
│       └── ci.yml
│
├── api/
│   ├── app.py
│   ├── Dockerfile
│   └── requirements.txt
│
├── frontend/
│   ├── frontend.py
│   ├── Dockerfile
│   └── requirements.txt
│
├── monitoring/
│   ├── monitoring.py
│   ├── Dockerfile
│   └── requirements.txt
│
├── tests/
│   └── test_api.py
│
├── docs/
│   └── screenshots
│
├── train_model.py
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Security and Configuration

Credentials are not stored directly in the repository.

Environment variables are used for values such as:

- Database password
- Database host
- W&B API key

GitHub Actions uses GitHub repository secrets for protected credentials.

---

## Documentation Screenshots

Project screenshots are stored in the `docs/` folder.

Screenshots include evidence of:

- AWS EC2 deployment
- AWS RDS PostgreSQL
- FastAPI application
- Streamlit prediction frontend
- Monitoring dashboard
- W&B experiment tracking
- W&B Model Registry
- GitHub Actions CI
- Successful automated tests

---

## Technologies

- Python
- Pandas
- Scikit-learn
- FastAPI
- Streamlit
- PostgreSQL
- AWS EC2
- AWS RDS
- Docker
- Weights & Biases
- GitHub Actions
- Pytest
- Ruff

---

## Repository

Public GitHub repository:

`https://github.com/drbsmooth27/finalProject-flightPredictions`

---

## Weights & Biases

Public W&B project:

https://wandb.ai/leo-dixon-university-of-denver/flight-delay-mlops/workspace?nw=nwuserleodixon
