from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import joblib
import json
import numpy as np
from datetime import datetime
from typing import List, Optional
import os
import psycopg2
from psycopg2.extras import RealDictCursor

# Initialize FastAPI
app = FastAPI(
    title="Credit Score Prediction API",
    description="API for predicting credit score and loan approval",
    version="1.0.0"
)
@app.get("/")
def home():
    return {"message": "Credit AI API is running"}
# CORS Configuration
origins = [
    "http://localhost:3000",
    "http://localhost:8000",
    "https://your-frontend.vercel.app",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "credit_model")

try:
    xgb_model = joblib.load(os.path.join(MODEL_DIR, "xgb_model.pkl"))
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))

    with open(os.path.join(MODEL_DIR, "feature_names.json")) as f:
        feature_names = json.load(f)

    with open(os.path.join(MODEL_DIR, "metadata.json")) as f:
        model_metadata = json.load(f)

    MODEL_LOADED = True
    print("Model loaded successfully")

except Exception as e:
    print(f"Error loading model: {e}")
    MODEL_LOADED = False

# Database Connection
def get_db_connection():
    try:
        conn = psycopg2.connect(
            host=os.getenv("DB_HOST"),
            database=os.getenv("DB_NAME"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
            port=os.getenv("DB_PORT", "5432")
        )
        return conn
    except Exception as e:
        print(f"Database connection error: {e}")
        return None

# Pydantic Models
class PredictRequest(BaseModel):
    income: float = Field(..., gt=0, description="Annual income in USD")
    age: int = Field(..., ge=18, le=100, description="Age of applicant")
    employment_years: int = Field(..., ge=0, le=60, description="Years of employment")
    loan_amount: float = Field(..., gt=0, description="Loan amount requested")
    loan_term: int = Field(..., ge=12, le=360, description="Loan term in months")
    credit_history_length: int = Field(..., ge=0, le=80, description="Credit history years")
    num_credit_lines: int = Field(..., ge=0, le=50, description="Number of credit lines")
    num_delinquencies: int = Field(..., ge=0, le=20, description="Number of delinquencies")
    debt_to_income_ratio: float = Field(..., ge=0, le=1, description="Debt to income ratio")
    savings_balance: float = Field(..., ge=0, description="Savings balance")

class PredictResponse(BaseModel):
    approval_score: float
    approved: bool
    risk_level: str
    recommendation: str
    timestamp: str

class ApplicationRecord(BaseModel):
    id: str
    input_data: dict
    approval_score: float
    approved: bool
    risk_level: str
    recommendation: str
    created_at: str

# Endpoints
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": MODEL_LOADED,
        "model_info": model_metadata if MODEL_LOADED else None
    }

@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    """Predict credit score and approval"""
    if not MODEL_LOADED:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Prepare input
        input_data = [
            request.income,
            request.age,
            request.employment_years,
            request.loan_amount,
            request.loan_term,
            request.credit_history_length,
            request.num_credit_lines,
            request.num_delinquencies,
            request.debt_to_income_ratio,
            request.savings_balance
        ]
        
        # Scale input
        X_scaled = scaler.transform([input_data])
        
        # Predict
        prediction = xgb_model.predict(X_scaled)[0]
        probability = xgb_model.predict_proba(X_scaled)[0][1]
        
        # Calculate approval score (0-100)
        approval_score = round(probability * 100, 2)
        approved = bool(prediction == 1)
        
        # Determine risk level
        if approval_score >= 80:
            risk_level = "Low"
        elif approval_score >= 60:
            risk_level = "Medium"
        else:
            risk_level = "High"
        
        # Recommendation
        if approved:
            recommendation = f"✓ Approved. Applicant has {approval_score:.0f}% approval score. Low risk profile."
        else:
            recommendation = f"✗ Rejected. Applicant has {approval_score:.0f}% approval score. Please review income and delinquency history."
        
        # Save to database
        conn = get_db_connection()
        if conn:
            try:
                cur = conn.cursor()
                cur.execute("""
                    INSERT INTO applications (input_data, approval_score, approved, risk_level, recommendation)
                    VALUES (%s, %s, %s, %s, %s)
                """, (
                    json.dumps(request.dict()),
                    approval_score,
                    approved,
                    risk_level,
                    recommendation
                ))
                conn.commit()
                cur.close()
            except Exception as e:
                print(f"Database insert error: {e}")
            finally:
                conn.close()
        
        return PredictResponse(
            approval_score=approval_score,
            approved=approved,
            risk_level=risk_level,
            recommendation=recommendation,
            timestamp=datetime.now().isoformat()
        )
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/applications")
async def get_applications(skip: int = 0, limit: int = 10):
    """Get application history with pagination"""
    conn = get_db_connection()
    if not conn:
        raise HTTPException(status_code=503, detail="Database unavailable")
    
    try:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute("""
            SELECT id, input_data, approval_score, approved, risk_level, recommendation, created_at
            FROM applications
            ORDER BY created_at DESC
            LIMIT %s OFFSET %s
        """, (limit, skip))
        
        records = cur.fetchall()
        cur.execute("SELECT COUNT(*) as total FROM applications")
        total = cur.fetchone()['total']
        
        return {
            "total": total,
            "skip": skip,
            "limit": limit,
            "data": records
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        cur.close()
        conn.close()

@app.get("/applications/{application_id}")
async def get_application(application_id: str):
    """Get single application details"""
    conn = get_db_connection()
    if not conn:
        raise HTTPException(status_code=503, detail="Database unavailable")
    
    try:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute("""
            SELECT id, input_data, approval_score, approved, risk_level, recommendation, created_at
            FROM applications
            WHERE id = %s
        """, (application_id,))
        
        record = cur.fetchone()
        if not record:
            raise HTTPException(status_code=404, detail="Application not found")
        
        return record
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        cur.close()
        conn.close()

@app.get("/model-info")
async def get_model_info():
    """Get model information"""
    if not MODEL_LOADED:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_type": model_metadata.get("model_type"),
        "accuracy": model_metadata.get("test_accuracy"),
        "roc_auc": model_metadata.get("roc_auc"),
        "features": model_metadata.get("feature_names")
    }
@app.get("/test-db")
def test_db():
    conn = psycopg2.connect(
        "postgresql://postgres:[YOUR-PASSWORD]@db.ronjoxkhmehifdbzyqtk.supabase.co:5432/postgres"
    )
    conn.close()
    return {"message": "Database connected"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
