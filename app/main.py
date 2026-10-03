from pathlib import Path
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from .risk_engine import score_order
from .data import ORDERS
import io, csv

BASE = Path(__file__).resolve().parent
app = FastAPI(title="CODShield API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class LoginRequest(BaseModel):
    email: str
    password: str

class DecisionRequest(BaseModel):
    decision: str

@app.get("/")
def root():
    return FileResponse(BASE / "static" / "index.html")

@app.post("/api/login")
def login(payload: LoginRequest):
    if payload.email == "demo@codshield.ai" and payload.password == "Demo123!":
        return {"ok": True, "token": "demo-session-token"}
    raise HTTPException(status_code=401, detail="Invalid demo credentials")

@app.get("/api/orders")
def get_orders():
    return {"orders": [score_order(o) for o in ORDERS]}

@app.get("/api/orders/{order_id}")
def get_order(order_id: str):
    order = next((o for o in ORDERS if o["id"] == order_id), None)
    if not order:
        raise HTTPException(status_code=404, detail="Order not found")
    return score_order(order)

@app.post("/api/orders/{order_id}/decision")
def set_decision(order_id: str, payload: DecisionRequest):
    if payload.decision not in {"dispatch", "review", "flagged"}:
        raise HTTPException(status_code=400, detail="Invalid decision")
    order = next((o for o in ORDERS if o["id"] == order_id), None)
    if not order:
        raise HTTPException(status_code=404, detail="Order not found")
    order["decision"] = payload.decision
    return {"ok": True, "order_id": order_id, "decision": payload.decision}

@app.post("/api/upload")
async def upload_csv(file: UploadFile = File(...)):
    """Accept a CSV of orders, score each row and return results."""
    content = await file.read()
    text = content.decode("utf-8-sig")
    reader = csv.DictReader(io.StringIO(text))
    results = []
    REQUIRED = {"id","customer","city","amount","previous_orders",
                "successful_deliveries","rto_count","address_similarity",
                "phone_reuse","delivery_zone_risk","product","hour"}
    for i, row in enumerate(reader):
        missing = REQUIRED - set(row.keys())
        if missing:
            raise HTTPException(
                status_code=422,
                detail=f"Row {i+1} missing columns: {missing}"
            )
        try:
            order = {
                "id": row["id"] or f"UP-{i+1:04d}",
                "customer": row["customer"],
                "city": row["city"],
                "amount": float(row["amount"]),
                "previous_orders": int(row["previous_orders"]),
                "successful_deliveries": int(row["successful_deliveries"]),
                "rto_count": int(row["rto_count"]),
                "address_similarity": float(row["address_similarity"]),
                "phone_reuse": int(row["phone_reuse"]),
                "delivery_zone_risk": float(row["delivery_zone_risk"]),
                "product": row["product"],
                "hour": int(row["hour"]),
                "decision": row.get("decision", "pending"),
            }
            results.append(score_order(order))
        except (ValueError, KeyError) as exc:
            raise HTTPException(status_code=422, detail=f"Row {i+1} parse error: {exc}")
    return {"count": len(results), "orders": results}

@app.get("/api/sample-csv")
def sample_csv():
    """Return a downloadable sample CSV template."""
    from fastapi.responses import Response
    header = "id,customer,city,amount,previous_orders,successful_deliveries,rto_count,address_similarity,phone_reuse,delivery_zone_risk,product,hour,decision\n"
    row = "CS-99001,Test Customer,Karachi,5500,3,2,1,0.62,1,0.55,Sneakers,14,pending\n"
    return Response(content=header+row, media_type="text/csv",
                    headers={"Content-Disposition": "attachment; filename=codshield_sample.csv"})
