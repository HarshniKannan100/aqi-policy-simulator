from fastapi import FastAPI
import joblib
import numpy as np
import requests

app = FastAPI()

# ---------- Load impact model ----------
impact_model = joblib.load("impact_model.pkl")

features = [
    "pm25","pm10","no2","so2","co","o3",
    "temperature","humidity","wind_speed"
]

WAQI_TOKEN = "5d884a451880e821b8e4c7ed3a8727ce0eb30650"
WAQI_URL = f"https://api.waqi.info/feed/delhi/?token={WAQI_TOKEN}"

SOURCE_MODEL_URL = "https://cirealkiller-source-identification-waqi.hf.space/live"


# ---------- Fetch WAQI ----------
def fetch_waqi():

    data = requests.get(WAQI_URL, timeout=5).json()
    iaqi = data["data"]["iaqi"]

    pollutants = {
        "pm25": iaqi.get("pm25",{}).get("v",0),
        "pm10": iaqi.get("pm10",{}).get("v",0),
        "no2": iaqi.get("no2",{}).get("v",0),
        "so2": iaqi.get("so2",{}).get("v",0),
        "co": iaqi.get("co",{}).get("v",0),
        "o3": iaqi.get("o3",{}).get("v",0),
        "temperature": iaqi.get("t",{}).get("v",25),
        "humidity": iaqi.get("h",{}).get("v",50),
        "wind_speed": iaqi.get("w",{}).get("v",2),
    }

    current_aqi = data["data"]["aqi"]

    return pollutants, current_aqi


# ---------- Get source attribution ----------
def get_sources():

    res = requests.get(SOURCE_MODEL_URL, timeout=5)
    data = res.json()

    probs = data.get("probabilities", {})

    return {
        "traffic": probs.get("traffic",0),
        "construction": probs.get("construction",0),
        "road_dust": probs.get("road_dust",0),
        "industry": probs.get("industry",0),
        "stubble": probs.get("stubble",0)
    }


# ---------- Sensitivity weights ----------
def get_weights():
    coeffs = impact_model.coef_
    total = np.sum(np.abs(coeffs))
    weights = np.abs(coeffs)/total
    return dict(zip(features,weights))


# ---------- Simulator ----------
def simulate(pollutants, source_percent, weights):

    sim = pollutants.copy()

    for f in features:
        reduction = (source_percent/100)*weights[f]
        sim[f] *= (1-reduction)

    X = np.array([[sim[f] for f in features]])
    change = impact_model.predict(X)[0]

    # safety cap
    change = max(change, -60)

    return change


# ---------- Endpoint ----------
@app.get("/policy-impact")
def policy():

    pollutants, current_aqi = fetch_waqi()
    sources = get_sources()
    weights = get_weights()

    results = {}

    # simulate all sources
    for source,percent in sources.items():

        change = simulate(pollutants, percent, weights)

        results[source] = {
            "aqi_change": change,
            "estimated_aqi": current_aqi + change
        }

    # rank policies (largest improvement first)
    ranked = sorted(results.items(), key=lambda x: x[1]["aqi_change"])

    best_source, best_result = ranked[0]

    return {
        "pollutants": pollutants,
        "sources": sources,
        "weights": weights,

        "policy_results": results,

        "top_recommendation": {
            "policy": best_source,
            "expected_change": best_result["aqi_change"],
            "estimated_aqi": best_result["estimated_aqi"]
        }
    }