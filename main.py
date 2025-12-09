import io
import pandas as pd
import joblib
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os

from pymongo import MongoClient
from dotenv import load_dotenv

from model_utils2 import compute_rule_columns

load_dotenv()   

MONGO_URI = os.getenv("MONGO_URI")  
DB_NAME = os.getenv("DB_NAME", "sih_dropout_db")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "predictions")


app = FastAPI(title="Dropout Prediction System - SIH")


ml_model = joblib.load("dropout_model_xgb.pkl")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

client = MongoClient(MONGO_URI)
db = client[DB_NAME]
collection = db[COLLECTION_NAME]


def compute_fee_weight(months_unpaid):
    if months_unpaid >= 3:
        return 1.0
    elif months_unpaid == 2:
        return 0.7
    elif months_unpaid == 1:
        return 0.4
    return 0.0



def get_zone(final_score):
    if final_score >= 65:
        return "Red Zone"
    elif final_score >= 25:
        return "Yellow Zone"
    return "Green Zone"



@app.post("/predict/batch")
async def batch_predict(
    attendance_csv: UploadFile = File(...),
    cgpa_csv: UploadFile = File(...),
    fees_csv: UploadFile = File(...)
):
    try:
        df_att = pd.read_csv(io.BytesIO(await attendance_csv.read()))
        df_cgpa = pd.read_csv(io.BytesIO(await cgpa_csv.read()))
        df_fees = pd.read_csv(io.BytesIO(await fees_csv.read()))


        if "fee_weight" not in df_fees.columns:
            df_fees["fee_weight"] = df_fees["outstanding_months"].apply(compute_fee_weight)


        df = df_att.merge(df_cgpa, on="student_id", how="left")
        df = df.merge(df_fees, on="student_id", how="left")


        if "avg_cgpa" not in df.columns:
            cgpa_cols = [c for c in df.columns if c.startswith("cgpa_sem")]
            if len(cgpa_cols) == 0:
                return {"error": "No columns found like cgpa_sem1, cgpa_sem2…"}
            df["avg_cgpa"] = df[cgpa_cols].mean(axis=1)


        df = compute_rule_columns(df)


        ml_features = ["attendance", "avg_cgpa", "fee_weight", "rule_score"]

       
        for col in ml_features:
            if col not in df.columns:
                df[col] = 0.0

        df["ml_pred"] = ml_model.predict_proba(df[ml_features].fillna(0))[:, 1]


        df["final_score"] = ((df["ml_pred"] + df["rule_score"]) / 2) * 100
        df["risk_zone"] = df["final_score"].apply(get_zone)

        records = df.to_dict(orient="records")
        collection.insert_many(records)
        print(f"Saved {len(records)} records to MongoDB Atlas")


        return {
            "total_records": len(df),
            "preview": df.head(10).to_dict(orient="records"),
            "full_data": df.to_dict(orient="records")
        }

    except Exception as e:
        return {"error": str(e)}



if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
