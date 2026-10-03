# CODShield MVP

AI-powered COD risk intelligence demo for Pakistani e-commerce.

## Demo credentials
- Email: demo@codshield.ai
- Password: Demo123!

## Run locally

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
# source .venv/bin/activate

pip install -r requirements.txt
uvicorn app.main:app --reload
```

Open http://127.0.0.1:8000

## What the MVP demonstrates

- Merchant login
- Dashboard KPIs
- Recent COD orders
- Individual order risk scoring
- Explainable risk factors
- Manual review / dispatch actions
- Synthetic order data
- ML-style risk engine using a trained Logistic Regression model
- REST API via FastAPI

This is a product demonstration using synthetic data; it is not a production fraud/RTO system.
