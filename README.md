# Customer Churn Prediction

**GitHub Repository**: https://github.com/NJVinay/Customer-Churn-Prediction

End-to-end churn prediction project with an ensemble model, FastAPI backend, and a static frontend.

## Project Overview

This project evaluates multiple models and demonstrates that an ensemble improves performance over single models. It includes reproducible benchmarking, a cost-based threshold policy, and a deployable API.

## Features

- Ensemble stacking model (Logistic Regression, Decision Tree, LightGBM, CatBoost)
- SMOTE for class imbalance
- Cost-based threshold selection
- Benchmarking script with ROC-AUC and PR-AUC
- FastAPI inference service

## Project Structure

```
Customer-Churn-Prediction/
  backend/                  # API, training, evaluation
  frontend/                 # Static UI
  data/                     # Dataset (Telco churn)
  reports/                  # Benchmark outputs (generated)
```

## Getting Started

### Backend (FastAPI)

1. Install dependencies:

```bash
pip install -r backend/requirements.txt
```

2. Train and save the ensemble model (cost-based threshold):

```bash
python backend/train_model.py
```

3. Run the API:

```bash
uvicorn backend.app:app --reload
```

### Benchmarking

Run baselines vs ensemble and save metrics to `reports/` (CV + holdout):

```bash
python backend/evaluate_models.py
```

### Frontend (static)

The frontend is a static site in `frontend/`.

1. Update `frontend/config.js` with your API URL.
2. Deploy the `frontend/` folder to any static host (or run locally with any static server).

### Deployment (Render)

Use `render.yaml` to deploy both services:

- **Backend**: Render Web Service (`uvicorn backend.app:app`)
- **Frontend**: Render Static Site (`frontend/`)

## License

This project is open source and available under the MIT License.
