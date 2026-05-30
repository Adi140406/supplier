import os
import json
import joblib
import pandas as pd
import numpy as np
from flask import Flask, render_template, jsonify, request
from flask_cors import CORS

app = Flask(__name__, template_folder="templates", static_folder="static")
CORS(app)

# --- paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "BIW_BPillar_Supplier_Dataset.csv")
MODEL_DIR = os.path.join(BASE_DIR, "model_output")
MODEL_PATH = os.path.join(MODEL_DIR, "failure_model.joblib")
META_PATH = os.path.join(MODEL_DIR, "model_meta.json")
SCORES_PATH = os.path.join(MODEL_DIR, "supplier_scores.csv")

# Global variables to cache model and dataset info
model = None
metadata = None
supplier_scores = None
dataset_stats = {}

def load_assets():
    global model, metadata, supplier_scores, dataset_stats
    
    # 1. Load trained model pipeline
    if os.path.exists(MODEL_PATH):
        try:
            model = joblib.load(MODEL_PATH)
            print(" -> Failure Prediction Model loaded successfully.")
        except Exception as e:
            print(f"Error loading model: {e}")
            
    # 2. Load model metadata
    if os.path.exists(META_PATH):
        try:
            with open(META_PATH, "r") as f:
                metadata = json.load(f)
            print(" -> Model Metadata loaded successfully.")
        except Exception as e:
            print(f"Error loading metadata: {e}")
            
    # 3. Load supplier scores & calculate extra metrics
    if os.path.exists(SCORES_PATH) and os.path.exists(DATA_PATH):
        try:
            supplier_scores = pd.read_csv(SCORES_PATH)
            df = pd.read_csv(DATA_PATH)
            
            # Map fail rates & counts from original dataset to enrich API
            df["fail"] = (df["Overall_Disposition"] == "REJECTED").astype(int)
            stats = df.groupby("Supplier_ID").agg(
                Fail_Rate=("fail", "mean"),
                Part_Count=("fail", "count"),
                Rejected_Count=("fail", "sum")
            ).reset_index()
            
            # Merge with existing score table
            supplier_scores = supplier_scores.merge(stats, on="Supplier_ID", how="left")
            supplier_scores["Fail_Rate"] = (supplier_scores["Fail_Rate"] * 100).round(2)
            print(" -> Supplier Scores loaded and enriched.")
            
            # Calculate standard medians of all features to fill user omissions
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for c in numeric_cols:
                dataset_stats[c] = float(df[c].median())
                
            # OK/NOT OK binary cols
            ok_cols = [c for c in df.columns if df[c].dtype == object
                       and set(df[c].dropna().unique()).issubset({"OK", "NOT OK"})]
            for c in ok_cols:
                dataset_stats[c + "_bin"] = 1.0  # Default to OK (1.0)
                
            dataset_stats["Supplier_Score"] = 75.0  # Average supplier score fallback
            
        except Exception as e:
            print(f"Error loading data assets: {e}")

load_assets()

# --- Server Routes ---

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/suppliers", methods=["GET"])
def get_suppliers():
    if supplier_scores is None:
        return jsonify({"error": "Supplier scores not loaded"}), 500
    
    # Return as list of dicts
    data = supplier_scores.to_dict(orient="records")
    return jsonify(data)

@app.route("/api/metadata", methods=["GET"])
def get_metadata():
    if metadata is None:
        return jsonify({"error": "Model metadata not loaded"}), 500
        
    # Dynamically extract feature coefficients if Logistic Regression is the model
    feature_importances = {}
    try:
        if model and hasattr(model, "named_steps") and "clf" in model.named_steps:
            clf = model.named_steps["clf"]
            if hasattr(clf, "coef_"):
                # Logistic Regression weights
                coefs = clf.coef_[0]
                feature_names = metadata.get("feature_cols", [])
                for name, val in zip(feature_names, coefs):
                    feature_importances[name] = float(val)
            elif hasattr(clf, "feature_importances_"):
                # Tree-based model importances
                importances = clf.feature_importances_
                feature_names = metadata.get("feature_cols", [])
                for name, val in zip(feature_names, importances):
                    feature_importances[name] = float(val)
    except Exception as e:
        print(f"Failed to extract coefficients: {e}")
        
    resp = {
        "best_model": metadata.get("best_model", "N/A"),
        "cv_roc_auc": round(metadata.get("cv_roc_auc_mean", 0), 4),
        "cv_f1": round(metadata.get("cv_f1_mean", 0), 4),
        "features": metadata.get("feature_cols", []),
        "feature_importances": feature_importances,
        "default_medians": dataset_stats
    }
    return jsonify(resp)

@app.route("/api/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Prediction model not loaded"}), 500
        
    try:
        payload = request.get_json(force=True) or {}
        feature_names = metadata.get("feature_cols", [])
        
        # Build features dict using defaults if missing in request payload
        row_dict = {}
        for feat in feature_names:
            if feat in payload:
                row_dict[feat] = float(payload[feat])
            else:
                row_dict[feat] = float(dataset_stats.get(feat, 0.0))
                
        # Handle custom supplier selection (map Supplier_ID to their Score)
        if "Supplier_ID" in payload and supplier_scores is not None:
            match = supplier_scores[supplier_scores["Supplier_ID"] == payload["Supplier_ID"]]
            if not match.empty:
                row_dict["Supplier_Score"] = float(match["Supplier_Score"].values[0])
        
        # Convert to Pandas DataFrame matching model's expected shape
        X_new = pd.DataFrame([row_dict], columns=feature_names)
        
        # Perform prediction
        pred = int(model.predict(X_new)[0])
        prob = float(model.predict_proba(X_new)[0][1])
        
        verdict = "FAIL" if pred == 1 else "PASS"
        
        return jsonify({
            "status": "success",
            "prediction": pred,
            "fail_probability": prob,
            "verdict": verdict,
            "features_used": row_dict
        })
        
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

if __name__ == "__main__":
    # Create templates and static directories if they don't exist
    os.makedirs(os.path.join(BASE_DIR, "templates"), exist_ok=True)
    os.makedirs(os.path.join(BASE_DIR, "static"), exist_ok=True)
    
    app.run(host="127.0.0.1", port=5000, debug=True)
