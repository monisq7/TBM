import streamlit as st
import certifi
import pandas as pd
from datetime import datetime, timezone
from pymongo import MongoClient
from pymongo.server_api import ServerApi


@st.cache_resource
def get_mongo_collection():
    """
    Creates a cached MongoDB client (Streamlit keeps this alive across reruns,
    so we don't reconnect on every button click).
    Reads the connection string from Streamlit secrets, NOT hardcoded here.

    tlsCAFile=certifi.where() fixes the common macOS/Python
    'CERTIFICATE_VERIFY_FAILED: unable to get local issuer certificate' error,
    which happens because Python's bundled OpenSSL doesn't always have access
    to the system's root certificate store.
    """
    uri = st.secrets["mongo"]["uri"]
    client = MongoClient(
        uri,
        server_api=ServerApi("1"),
        tlsCAFile=certifi.where()
    )
    db = client[st.secrets["mongo"].get("db_name", "tbm_predictor")]
    return db[st.secrets["mongo"].get("collection_name", "predictions")]


def save_prediction(input_data: pd.DataFrame, death_prob: float, risk: str, patient_name: str = ""):
    """
    Saves one prediction record to MongoDB. Wrapped in try/except so that
    if MongoDB is unreachable, the app still shows the prediction to the user
    instead of crashing.

    Returns (success: bool, error_message: str | None)
    """
    try:
        collection = get_mongo_collection()
        record = {"patient_name": patient_name.strip() if patient_name else "Unnamed"}
        record.update(input_data.iloc[0].to_dict())
        record["death_probability_percent"] = round(death_prob, 2)
        record["risk_category"] = risk
        record["timestamp_utc"] = datetime.now(timezone.utc)
        collection.insert_one(record)
        return True, None
    except Exception as e:
        return False, str(e)