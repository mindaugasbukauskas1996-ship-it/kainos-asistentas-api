from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import re
import csv

app = FastAPI(title="Kainos asistentas API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # MVP – leidžiam iš visur (vėliau apribosim iki tavo GitHub Pages)
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

    # tarblokinės siūlės (fasadas)
    if any(x in t for x in ["tarblokin", "tarpblok", "tarp blok", "siūl", "siuli", "siūlių", "siuliu"]) and any(
        x in t for x in ["fasad", "siena", "tarpblok", "tarblokin"]
    ):
        return "FACADE_SEAM"

    # nuotekos / kanalizacija
    if any(x in t for x in ["nuotek", "kanaliz", "kanalizacij"]):
        return "SEWER"

    # stovai / vamzdžiai
    if "stov" in t:
        return "PIPE_STACK"
    if "vamzd" in t:
        return "PIPE"

    # stogas / čerpės / danga
    if any(x in t for x in ["stog", "čerpi", "cerpi", "danga"]):
        return "ROOF"

    if any(x in t for x in ["šviest", "lemput", "apsviet", "apšviet", "elektr"]):
        return "LIGHT"
    if any(x in t for x in ["radiator", "nuorin"]):
        return "RADIATOR"
    if any(x in t for x in ["spyn", "dur", "pritrauk"]):
        return "LOCK_DOOR"

    return "OTHER"
    t = (text or "").lower()

    # tarblokinės siūlės (fasadas)
    if any(x in t for x in ["tarblokin", "tarp blok", "tarpblok", "siūl", "siuli", "siūlių", "siuliu"]) and any(
        x in t for x in ["fasad", "siena", "tarpblok", "tarblokin", "tarblok"]
    ):
        return "FACADE_SEAM"

    # nuotekos / kanalizacija
    if any(x in t for x in ["nuotek", "kanaliz", "kanalizacij"]):
        return "SEWER"

    # stovai / vamzdžiai
    if any(x in t for x in ["stov"]):
        return "PIPE_STACK"
    if any(x in t for x in ["vamzd"]):
        return "PIPE"

    # stogas / čerpės / danga
    if any(x in t for x in ["stog", "čerpi", "cerpi", "danga"]):
        return "ROOF"

    if any(x in t for x in ["šviest", "lemput", "apsviet", "apšviet", "elektr"]):
        return "LIGHT"
    if any(x in t for x in ["radiator", "nuorin"]):
        return "RADIATOR"
    if any(x in t for x in ["spyn", "dur", "pritrauk"]):
        return "LOCK_DOOR"

    return "OTHER"


def extract_quantity(text: str):
    """
    Returns (qty, unit) where unit in {"m","m2","vnt","aukstas"} or (None,None).
    """
    t = (text or "").lower()

    # avoid years like "2025 m."
    t = re.sub(r"\b(19\d{2}|20\d{2})\s*m\.\b", r"\1 metai", t)

    # Normalize m² to m2
    t = t.replace("m²", "m2").replace("㎡", "m2")

    patterns = [
        (r"(\d+(?:[.,]\d+)?)\s*(m2)\b", "m2"),
        (r"(\d+(?:[.,]\d+)?)\s*(m)\b", "m"),
        (r"(\d+(?:[.,]\d+)?)\s*(vnt)\b", "vnt"),
        (r"(\d+(?:[.,]\d+)?)\s*(aukšt(?:as|ų)?)\b", "aukstas"),
    ]

    matches = []
    for rgx, unit in patterns:
        for m in re.finditer(rgx, t):
            num = m.group(1).replace(",", ".")
            try:
                val = float(num)
            except:
                continue
            if unit == "m" and 1900 <= val <= 2100:
                continue
            matches.append((val, unit))

    # prefer m2 then m then aukstas then vnt
    order = {"m2": 0, "m": 1, "aukstas": 2, "vnt": 3}
    if matches:
        matches.sort(key=lambda x: (order.get(x[1], 99), -x[0]))
        return matches[0][0], matches[0][1]

    return None, None


def has_trisakis(text: str) -> bool:
    t = (text or "").lower()
    return ("trišak" in t) or ("trisak" in t)


def water_type(text: str) -> str:
    """
    sewer / hot / cold / unknown
    """
    t = (text or "").lower()
    if any(x in t for x in ["nuotek", "kanaliz"]):
        return "sewer"
    if any(x in t for x in ["karšto", "karstas", "karšto vandens"]):
        return "hot"
    if any(x in t for x in ["šalto", "saltas", "šalto vandens"]):
        return "cold"
    return "unknown"


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

    # Questions if missing qty
    followups = []
    t_low = text.lower()

    if "stov" in t_low and qty is None:
        followups.append("Kiek aukštų keičiamas stovas (pvz., 1, 2, 3 aukštai)?")
        followups.append("Ar su trišakiu (taip/ne)?")
    elif work_type in {"PIPE", "SEWER"} and qty is None:
        followups.append("Kiek metrų (m) vamzdžio/stovo reikia keisti ar remontuoti?")
        followups.append("Ar tai nuotekos, ar karšto/šalto vandens vamzdis?")
 elif work_type == "ROOF" and qty is None:
    followups.append("Kiek m² stogo remontuojama? (pvz. 12 m2)")
    followups.append("Jei čerpės – ar skaičiuojam vnt ar m²?")
elif work_type == "FACADE_SEAM" and qty is None:
    followups.append("Kiek metrų (m) tarblokinės siūlės bus sandarinama? (pvz. 25 m)")
    elif work_type in {"LIGHT", "RADIATOR", "LOCK_DOOR"} and qty is None:
        followups.append("Kiek vienetų (vnt) reikia keisti / sutvarkyti?")

    if followups:
        return {
            "status": "need_more_info",
            "work_type_guess": work_type,
            "questions": followups[:3],
        }

    if qty is not None and unit is None:
        return {
            "status": "need_more_info",
            "work_type_guess": work_type,
            "questions": ["Nurodykite vienetą: m, m², vnt arba aukštai."],
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

    # coefficients
    wtype = water_type(text)
    coef = 1.0
    if work_type == "SEWER":
        coef = 0.85
    elif work_type == "PIPE":
        if wtype == "hot":
            coef = 1.10
        elif wtype == "cold":
            coef = 1.05
        elif wtype == "sewer":
            coef = 0.85

    # trišakis add-on for stoves
    trisakis_add = 0.0
    if "stov" in t_low and has_trisakis(text):
        trisakis_add = 60.0

    est = qty * median * coef + trisakis_add
    low = qty * p25 * coef + trisakis_add
    high = qty * p75 * coef + trisakis_add

    return {
        "status": "ok",
        "work_type": work_type,
        "qty": qty,
        "unit": unit,
        "coef": coef,
        "trisakis_add": trisakis_add,
        "estimate_eur_be_pvm": round(est, 2),
        "range_eur_be_pvm": [round(low, 2), round(high, 2)],
        "assumptions": {
            "water_type": wtype,
        },
    }



