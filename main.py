from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from alert import send_email, send_telegram, send_sms
import os
from dotenv import load_dotenv
from removed_cols import cols_tobe_dropped

app = FastAPI()

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load models
rf_binary = joblib.load("rf_binary.pkl")
rf_mc = joblib.load("rf_multi.pkl")
dnn_binary = tf.keras.models.load_model("dnn_binary.keras")
dnn_mc = tf.keras.models.load_model("dnn_multi.keras")
cnn_binary = tf.keras.models.load_model("cnn_binary.keras")
cnn_mc = tf.keras.models.load_model("cnn_multi.keras")

# Global state tracker
row_index = 0

# Dataset and preprocessing
path = "C:/Users/admin/Documents/Intern/Projects/datasets/CICIDS/selected_features.csv"
df = pd.read_csv(path)
df = df.drop(columns=cols_tobe_dropped, errors='ignore')
df = df.head(30)

# Clean and encode
for col in df.columns:
    df[col] = df[col].replace([np.inf, -np.inf], np.nan)
    if df[col].dtypes == 'object':
        df[col].fillna(df.mode()[0], inplace=True)
    else:
        df[col].fillna(df.mean(), inplace=True)

label_encoders = {}
for col in df.columns:
    if df[col].dtypes == "object":
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le

# Reshape for CNN
def reshape_to_image(X, h=6, w=6):
    padded = np.pad(X, ((0, 0), (0, h * w - X.shape[1])), 'constant')
    return padded.reshape(-1, h, w, 1)

# Decode predictions
def reshape_predictions(preds, label):
    return [label[int(np.argmax(p))] for p in preds]

class_names = ['Bot', 'Brute Force', 'DDoS', 'DoS', 'Exploits', 'Normal Traffic', 'PortScan', 'Web Attack']
class_name = ['Normal Traffic', 'Attack']

@app.get("/api/new-predict")
async def predict():
    global row_index
    try:
        if row_index >= len(df):
            return JSONResponse(content={"message": "All rows have been predicted."})

        row = df.iloc[[row_index]]  # Keep it as DataFrame
        cnn_input = reshape_to_image(row)

        rf_binary_preds = [class_name[i] for i in rf_binary.predict(row)]
        rf_mc_preds = [class_names[i] for i in rf_mc.predict(row)]
        dnn_binary_preds = reshape_predictions(dnn_binary.predict(row), class_name)
        dnn_mc_preds = reshape_predictions(dnn_mc.predict(row), class_names)
        cnn_binary_preds = reshape_predictions(cnn_binary.predict(cnn_input), class_name)
        cnn_mc_preds = reshape_predictions(cnn_mc.predict(cnn_input), class_names)
        row_index += 1

        if ('Attack' in rf_binary_preds) or ('Attack' in dnn_binary_preds) or ('Attack' in cnn_binary_preds):
            # Load environment variables from .env file
            load_dotenv()
            send_email(os.getenv("SUBJECT"), os.getenv("BODY"), os.getenv("TO_EMAIL"))
            send_telegram(os.getenv("SUBJECT") + "\n" + os.getenv("BODY"), os.getenv("BOT_TOKEN"), os.getenv("CHAT_ID"))
            send_sms(
                message=os.getenv("SUBJECT") + "\n" + os.getenv("BODY"),
                to_number=os.getenv("TO"),
                account_sid=os.getenv("ACCOUNT_SID"),
                auth_token=os.getenv("AUTH_TOKEN"),
                from_number=os.getenv("FROM")
            )
            
        return JSONResponse(
            content={
                "row": int(row_index),
                "rf_binary_preds": rf_binary_preds,
                "rf_multi_preds": rf_mc_preds,
                "dnn_binary_preds": dnn_binary_preds,
                "dnn_multi_preds": dnn_mc_preds,
                "cnn_binary_preds": cnn_binary_preds,
                "cnn_multi_preds": cnn_mc_preds,
            }
        )

    except Exception as e:
        return JSONResponse(
            status_code=400,
            content={"error": f"An error occurred: {str(e)}"}
        )
       