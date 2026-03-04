from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import csv
import re
import json
import os
import math
from collections import Counter

app = FastAPI(title="Kainos asistentas API")

# ---------------- CORS ----------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://mindaugasbukauskas1996-ship-it.github.io"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- Load price table ----------------

PRICE = []

with open("price_table.csv", newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)

    for r in reader:
        PRICE.append({
            "work_type": (r["work_type"] or "").upper().strip(),
            "unit": (r["unit"] or "").lower().strip(),
            "median_unit_price": float(r["median_unit_price"]),
            "p25": float(r["p25"]),
            "p75": float(r["p75"]),
        })


# ---------------- Load ML model ----------------

MODEL = None
MODEL_PATH = "cluster_tfidf_model.json"

if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, "r", encoding="utf-8") as f:
        MODEL = json.load(f)


# ---------------- Tokenization ----------------

def tokenize_lt(text: str):

    t = (text or "").lower()

    t = re.sub(r"[^0-9a-ząčęėįšųūž]+", " ", t)

    toks = [x for x in t.split() if len(x) >= 2]

    return toks


# ---------------- ML cluster prediction ----------------

def predict_cluster(text: str):

    if not MODEL:
        return None, None, []

    idf = MODEL["idf"]

    vocab = set(idf.keys())

    toks = tokenize_lt(text)

    tf = Counter([t for t in toks if t in vocab])

    if not tf:
        return None, None, []

    vec = {}

    norm2 = 0

    for t, c in tf.items():

        w = (1 + math.log(c)) * float(idf[t])

        vec[t] = w

        norm2 += w * w

    norm = math.sqrt(norm2) if norm2 > 0 else 1

    for t in list(vec.keys()):
        vec[t] /= norm

    scores = []

    for cl, items in MODEL["centroids"].items():

        s = 0

        for tok, w in items:

            if tok in vec:
                s += vec[tok] * float(w)

        scores.append((cl, s))

    scores.sort(key=lambda x: x[1], reverse=True)

    best = scores[0] if scores else (None, None)

    top3 = scores[:3]

    return best[0], best[1], top3


# ---------------- Rule fallback ----------------

def detect_work_type(text: str):

    t = (text or "").lower()

    if "tarblokin" in t or "tarpblok" in t:
        return "FACADE_SEAM"

    if "nuotek" in t or "kanaliz" in t:
        return "SEWER"

    if "stov" in t:
        return "PIPE_STACK"

    if "vamzd" in t:
        return "PIPE"

    if "stog" in t or "čerpi" in t or "danga" in t:
        return "ROOF"

    if "šviest" in t or "lemput" in t:
        return "LIGHT"

    if "radiator" in t:
        return "RADIATOR"

    if "spyn" in t or "dur" in t:
        return "LOCK_DOOR"

    return "OTHER"


# ---------------- Quantity extraction ----------------

def extract_quantity(text: str):

    t = (text or "").lower()

    t = t.replace("m²", "m2")

    patterns = [
        (r"(\d+(?:[.,]\d+)?)\s*m2", "m2"),
        (r"(\d+(?:[.,]\d+)?)\s*m\b", "m"),
        (r"(\d+(?:[.,]\d+)?)\s*vnt", "vnt"),
        (r"(\d+(?:[.,]\d+)?)\s*aukšt", "aukstas"),
    ]

    for rgx, unit in patterns:

        m = re.search(rgx, t)

        if m:

            val = float(m.group(1).replace(",", "."))

            return val, unit

    return None, None


# ---------------- Helpers ----------------

def has_trisakis(text: str):

    t = (text or "").lower()

    return "trišak" in t or "trisak" in t


def get_price(work_type: str, unit: str):

    for r in PRICE:

        if r["work_type"] == work_type and r["unit"] == unit:

            return r

    return None


# ---------------- API schema ----------------

class EstimateRequest(BaseModel):

    text: str

    address: str | None = None


# ---------------- API ----------------

@app.post("/estimate")
def estimate(req: EstimateRequest):

    text = (req.text or "").strip()

    # ML prediction
    cluster_id, cluster_score, cluster_top3 = predict_cluster(text)

    # rule fallback
    work_type = detect_work_type(text)

    qty, unit = extract_quantity(text)

    followups = []

    if work_type == "PIPE_STACK" and qty is None:

        followups.append("Kiek aukštų keičiamas stovas? (pvz. 1 aukštas, 2 aukštai)")

        followups.append("Ar su trišakiu?")

    elif work_type in {"PIPE", "SEWER"} and qty is None:

        followups.append("Kiek metrų vamzdžio reikia keisti? (pvz. 6 m)")

    elif work_type == "ROOF" and qty is None:

        followups.append("Kiek m² stogo remontuojama? (pvz. 12 m2)")

    elif work_type == "FACADE_SEAM" and qty is None:

        followups.append("Kiek metrų tarblokinės siūlės sandarinama? (pvz. 25 m)")

    if followups:

        return {
            "status": "need_more_info",
            "work_type_guess": work_type,
            "questions": followups,
            "cluster": {
                "id": cluster_id,
                "score": cluster_score,
                "top3": cluster_top3
            }
        }

    price = get_price(work_type, unit)

    if price is None:

        return {
            "status": "no_price_model",
            "work_type_guess": work_type,
            "message": "Šiam darbui neturiu kainų modelio.",
            "cluster": {
                "id": cluster_id,
                "score": cluster_score,
                "top3": cluster_top3
            }
        }

    median = price["median_unit_price"]

    p25 = price["p25"]

    p75 = price["p75"]

    trisakis_add = 0

    if work_type == "PIPE_STACK" and has_trisakis(text):

        trisakis_add = 60

    est = qty * median + trisakis_add

    low = qty * p25 + trisakis_add

    high = qty * p75 + trisakis_add

    return {

        "status": "ok",

        "work_type": work_type,

        "qty": qty,

        "unit": unit,

        "estimate_eur_be_pvm": round(est, 2),

        "range_eur_be_pvm": [round(low, 2), round(high, 2)],

        "cluster": {
            "id": cluster_id,
            "score": cluster_score,
            "top3": cluster_top3
        }
    }
