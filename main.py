from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import csv
import re
import json
import os
import math
from collections import Counter, defaultdict
from typing import Optional, List, Dict, Any, Tuple

app = FastAPI(title="Kainos asistentas API")

# ---------------- CORS ----------------
# Leisti užklausas iš GitHub Pages UI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://mindaugasbukauskas1996-ship-it.github.io"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- Load price table ----------------
PRICE: List[Dict[str, Any]] = []

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

def get_price(work_type: str, unit: str) -> Optional[Dict[str, Any]]:
    for r in PRICE:
        if r["work_type"] == work_type and r["unit"] == unit:
            return r
    return None

# ---------------- Load historical jobs (for analogs) ----------------
HIST: List[Dict[str, Any]] = []
HIST_INDEX: Dict[Tuple[str, str], List[int]] = defaultdict(list)

HIST_PATH = "historical_jobs_min.csv"
if os.path.exists(HIST_PATH):
    with open(HIST_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, r in enumerate(reader):
            # Normalize
            wt = (r.get("work_type") or "").upper().strip()
            unit = (r.get("unit") or "").lower().strip()
            try:
                qty = float((r.get("qty_extracted") or "").replace(",", "."))
            except:
                qty = None
            try:
                cost = float((r.get("cost_be_pvm_eur") or "").replace(",", "."))
            except:
                cost = None
            try:
                up = float((r.get("unit_price") or "").replace(",", "."))
            except:
                up = None

            row = {
                "job_id": r.get("job_id"),
                "registration_nr": r.get("registration_nr_full"),
                "address": r.get("address_final"),
                "title": r.get("samatos_pavadinimas"),
                "contractor": r.get("contractor"),
                "work_type": wt,
                "unit": unit,
                "qty": qty,
                "cost_be_pvm": cost,
                "unit_price": up,
            }
            HIST.append(row)
            if wt and unit:
                HIST_INDEX[(wt, unit)].append(i)

def pick_analogs(work_type: str, unit: str, qty: Optional[float], top_k: int = 5) -> List[Dict[str, Any]]:
    idxs = HIST_INDEX.get((work_type, unit), [])
    if not idxs:
        return []

    price_row = get_price(work_type, unit)
    target_up = price_row["median_unit_price"] if price_row else None

    scored = []
    for i in idxs:
        r = HIST[i]
        up = r.get("unit_price")
        if up is None:
            continue
        # Primary: closeness to median unit price
        s = abs(up - target_up) if target_up is not None else 0.0
        # Secondary: closeness to qty (if we have qty)
        if qty is not None and r.get("qty") is not None:
            s += 0.05 * abs(r["qty"] - qty)
        scored.append((s, r))

    scored.sort(key=lambda x: x[0])
    out = []
    for _, r in scored[:top_k]:
        out.append({
            "registration_nr": r.get("registration_nr"),
            "address": r.get("address"),
            "title": r.get("title"),
            "contractor": r.get("contractor"),
            "qty": r.get("qty"),
            "unit": r.get("unit"),
            "cost_be_pvm": r.get("cost_be_pvm"),
            "unit_price": r.get("unit_price"),
        })
    return out

# ---------------- Load ML model ----------------
MODEL = None
MODEL_PATH = "cluster_tfidf_model.json"
if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, "r", encoding="utf-8") as f:
        MODEL = json.load(f)

def tokenize_lt(text: str) -> List[str]:
    t = (text or "").lower()
    t = re.sub(r"[^0-9a-ząčęėįšųūž]+", " ", t)
    return [x for x in t.split() if len(x) >= 2]

def predict_cluster(text: str):
    """
    Returns (best_cluster:str, best_score:float, top3:list[(cluster,score)])
    """
    if not MODEL:
        return None, None, []

    idf = MODEL.get("idf", {})
    if not idf:
        return None, None, []

    vocab = set(idf.keys())
    toks = tokenize_lt(text)
    tf = Counter([t for t in toks if t in vocab])
    if not tf:
        return None, None, []

    vec = {}
    norm2 = 0.0
    for t, c in tf.items():
        w = (1.0 + math.log(c)) * float(idf[t])
        vec[t] = w
        norm2 += w * w

    norm = math.sqrt(norm2) if norm2 > 0 else 1.0
    for t in list(vec.keys()):
        vec[t] /= norm

    scores = []
    centroids = MODEL.get("centroids", {})
    for cl, items in centroids.items():
        s = 0.0
        for tok, w in items:
            if tok in vec:
                s += vec[tok] * float(w)
        scores.append((cl, s))

    scores.sort(key=lambda x: x[1], reverse=True)
    best = scores[0] if scores else (None, None)
    return best[0], best[1], scores[:3]

# ---------------- Rule fallback (work_type) ----------------
def detect_work_type(text: str) -> str:
    t = (text or "").lower()

    # Tarblokinės / tarpblokinių siūlių sandarinimas (fasadas) – visada m
    if any(x in t for x in ["tarblokin", "tarpblok", "tarp blok"]):
        return "FACADE_SEAM"
    if any(x in t for x in ["siūl", "siuli", "siūlių", "siuliu"]) and any(x in t for x in ["fasad", "siena", "tarpblok", "tarblokin"]):
        return "FACADE_SEAM"

    if any(x in t for x in ["nuotek", "kanaliz", "kanalizacij"]):
        return "SEWER"

    # Stovai (€/aukštas)
    if "stov" in t:
        return "PIPE_STACK"

    # Vamzdžiai (€/m)
    if "vamzd" in t:
        return "PIPE"

    # Stogas (m2)
    if any(x in t for x in ["stog", "čerpi", "cerpi", "danga"]):
        return "ROOF"

    if any(x in t for x in ["šviest", "lemput", "apsviet", "apšviet", "elektr"]):
        return "LIGHT"

    if any(x in t for x in ["radiator", "nuorin"]):
        return "RADIATOR"

    if any(x in t for x in ["spyn", "dur", "pritrauk"]):
        return "LOCK_DOOR"

    return "OTHER"

# ---------------- Quantity extraction ----------------
def extract_quantity(text: str):
    """
    Returns (qty, unit) where unit in {"m","m2","vnt","aukstas"} or (None,None).
    """
    t = (text or "").lower()
    t = t.replace("m²", "m2")

    patterns = [
        (r"(\d+(?:[.,]\d+)?)\s*m2\b", "m2"),
        (r"(\d+(?:[.,]\d+)?)\s*m\b", "m"),
        (r"(\d+(?:[.,]\d+)?)\s*vnt\b", "vnt"),
        (r"(\d+(?:[.,]\d+)?)\s*aukšt", "aukstas"),
    ]

    for rgx, unit in patterns:
        m = re.search(rgx, t)
        if m:
            val = float(m.group(1).replace(",", "."))
            return val, unit

    return None, None

def has_trisakis(text: str) -> bool:
    t = (text or "").lower()
    return ("trišak" in t) or ("trisak" in t)

# ---------------- API schema ----------------
class EstimateRequest(BaseModel):
    text: str
    address: Optional[str] = None

# ---------------- API ----------------
@app.post("/estimate")
def estimate(req: EstimateRequest):
    text = (req.text or "").strip()

    # ML prediction (for debugging / future mapping)
    cluster_id, cluster_score, cluster_top3 = predict_cluster(text)

    # For now: keep stable rules for work_type (later we'll map cluster -> work_type)
    work_type = detect_work_type(text)

    qty, unit = extract_quantity(text)

    followups = []
    if work_type == "PIPE_STACK" and qty is None:
        followups.append("Kiek aukštų keičiamas stovas? (pvz. 1 aukštas, 2 aukštai, 3 aukštai)")
        followups.append("Ar su trišakiu? (taip/ne)")
    elif work_type in {"PIPE", "SEWER"} and qty is None:
        followups.append("Kiek metrų (m) reikia keisti / remontuoti? (pvz. 6 m)")
    elif work_type == "ROOF" and qty is None:
        followups.append("Kiek m² stogo remontuojama? (pvz. 12 m2)")
    elif work_type == "FACADE_SEAM" and qty is None:
        followups.append("Kiek metrų (m) tarblokinės siūlės sandarinama? (pvz. 25 m)")
    elif work_type in {"LIGHT", "RADIATOR", "LOCK_DOOR"} and qty is None:
        followups.append("Kiek vienetų (vnt) reikia keisti / sutvarkyti? (pvz. 2 vnt)")

    if followups:
        return {
            "status": "need_more_info",
            "work_type_guess": work_type,
            "questions": followups,
            "cluster": {"id": cluster_id, "score": cluster_score, "top3": cluster_top3},
        }

    if qty is None or unit is None:
        return {
            "status": "need_more_info",
            "work_type_guess": work_type,
            "questions": ["Nurodykite kiekį su vienetu: m, m2, vnt arba aukštai."],
            "cluster": {"id": cluster_id, "score": cluster_score, "top3": cluster_top3},
        }

    price = get_price(work_type, unit)
    if price is None:
        return {
            "status": "no_price_model",
            "work_type_guess": work_type,
            "message": "Šiam darbui neturiu pakankamai analogų kainai įvertinti. Reikia rangovo pasiūlymo arba tikslesnio darbo tipo.",
            "cluster": {"id": cluster_id, "score": cluster_score, "top3": cluster_top3},
        }

    median = float(price["median_unit_price"])
    p25 = float(price["p25"])
    p75 = float(price["p75"])

    trisakis_add = 0.0
    if work_type == "PIPE_STACK" and has_trisakis(text):
        trisakis_add = 60.0

    est = qty * median + trisakis_add
    low = qty * p25 + trisakis_add
    high = qty * p75 + trisakis_add

    analogs = pick_analogs(work_type, unit, qty, top_k=5)

    return {
        "status": "ok",
        "work_type": work_type,
        "qty": qty,
        "unit": unit,
        "trisakis_add": trisakis_add,
        "estimate_eur_be_pvm": round(est, 2),
        "range_eur_be_pvm": [round(low, 2), round(high, 2)],
        "analogs": analogs,
        "cluster": {"id": cluster_id, "score": cluster_score, "top3": cluster_top3},
    }
