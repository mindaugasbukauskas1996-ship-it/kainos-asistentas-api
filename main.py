from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import re
import csv

app = FastAPI(title="Kainos asistentas API")

# CORS (GitHub Pages UI)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://mindaugasbukauskas1996-ship-it.github.io"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Load price table ----------
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

# ---------- Helpers ----------
def detect_work_type(text: str) -> str:
    t = (text or "").lower()

    # Tarblokinės / tarpblokinių siūlių sandarinimas (fasadas) – visada m
    if any(x in t for x in ["tarblokin", "tarpblok", "tarp blok"]):
        return "FACADE_SEAM"
    if any(x in t for x in ["siūl", "siuli", "siūlių", "siuliu"]) and any(x in t for x in ["fasad", "siena", "tarpblok", "tarblokin"]):
        return "FACADE_SEAM"

    # Nuotekos / kanalizacija
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

    # Elektra / apšvietimas
    if any(x in t for x in ["šviest", "lemput", "apsviet", "apšviet", "elektr"]):
        return "LIGHT"

    # Radiatoriai
    if any(x in t for x in ["radiator", "nuorin"]):
        return "RADIATOR"

    # Spynos / durys
    if any(x in t for x in ["spyn", "dur", "pritrauk"]):
        return "LOCK_DOOR"

    return "OTHER"


def extract_quantity(text: str):
    """
    Returns (qty, unit) where unit in {"m","m2","vnt","aukstas"} or (None,None).
    """
    t = (text or "").lower()

    # Normalize m² to m2
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


def get_price(work_type: str, unit: str):
    for r in PRICE:
        if r["work_type"] == work_type and r["unit"] == unit:
            return r
    return None


# ---------- API schema ----------
class EstimateRequest(BaseModel):
    text: str
    address: str | None = None


@app.post("/estimate")
def estimate(req: EstimateRequest):
    text = (req.text or "").strip()
    work_type = detect_work_type(text)

    qty, unit = extract_quantity(text)
    followups = []

    if work_type == "PIPE_STACK" and qty is None:
        followups.append("Kiek aukštų keičiamas stovas? (pvz. 1 aukštas, 2 aukštai, 3 aukštai)")
        followups.append("Ar su trišakiu? (taip/ne)")

    elif work_type in {"PIPE", "SEWER"} and qty is None:
        followups.append("Kiek metrų (m) vamzdžio reikia keisti / remontuoti? (pvz. 6 m)")

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
        }

    price = get_price(work_type, unit)
    if price is None:
        return {
            "status": "no_price_model",
            "work_type_guess": work_type,
            "message": "Šiam darbui neturiu pakankamai analogų kainai įvertinti. Reikia rangovo pasiūlymo arba tikslesnio darbo tipo.",
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

    return {
        "status": "ok",
        "work_type": work_type,
        "qty": qty,
        "unit": unit,
        "trisakis_add": trisakis_add,
        "estimate_eur_be_pvm": round(est, 2),
        "range_eur_be_pvm": [round(low, 2), round(high, 2)],
    }
