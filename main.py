from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import os
import re
import statistics
import unicodedata

from openai_parser import parse_text
from rag_search import search_similar

app = FastAPI(title="Kainos asistentas API (RAG)")

# -------------------- Utils --------------------
def norm(s: str) -> str:
    s = (s or "").lower()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))  # ąčęėįšųūž -> aceeisuuz
    return s

def fallback_work_type(text: str) -> str:
    t = norm(text)

    # Tarplokinės / tarpblokinės siūlės fasade
    if (("tarplokin" in t) or ("tarpblokin" in t) or ("tarpblok" in t) or ("interpanel" in t)) and (
        ("siul" in t) or ("sandarin" in t) or ("hermet" in t)
    ):
        return "FACADE_SEAM"
    if ("siul" in t) and ("fasad" in t):
        return "FACADE_SEAM"

    # Stogai
    if ("stog" in t) or ("cerp" in t):
        return "PITCHED_ROOF_TILE"
    if ("bitum" in t) or ("ruberoid" in t) or ("ploksci" in t):
        return "FLAT_ROOF"

    # Stovai / vamzdynai
    if ("nuotek" in t) or ("kanaliz" in t):
        return "SEWER_STACK"
    if "stov" in t:
        return "WATER_STACK"
    if "vamzd" in t:
        return "PIPE"

    return "OTHER"

def pct(xs, p):
    if not xs:
        return None
    xs = sorted(xs)
    k = (len(xs) - 1) * p
    f = int(k)
    c = min(f + 1, len(xs) - 1)
    if f == c:
        return xs[f]
    return xs[f] + (xs[c] - xs[f]) * (k - f)

def calc_from_analogs(analogs: List[Dict[str, Any]], qty: float, unit: str):
    # Naudojam tik analogus su (cost/qty) kai unit sutampa
    ups = []
    unit_l = (unit or "").lower().strip()

    for a in analogs:
        aq = a.get("qty")
        ac = a.get("cost_be_pvm") or a.get("cost")  # jei rag_search grąžina cost
        au = (a.get("unit") or "").lower().strip()
        if aq and ac and au == unit_l:
            try:
                aq = float(aq)
                ac = float(ac)
                if aq > 0:
                    ups.append(ac / aq)
            except:
                pass

    # Leiskim skaičiuoti ir su mažiau analogų (bet bus platesnis intervalas)
    if not ups:
        return None

    med = statistics.median(ups)
    if len(ups) >= 3:
        p25 = pct(ups, 0.25)
        p75 = pct(ups, 0.75)
    elif len(ups) == 2:
        p25, p75 = min(ups), max(ups)
    else:  # len == 1
        p25, p75 = ups[0] * 0.75, ups[0] * 1.25

    return {
        "median_unit_price": round(med, 2),
        "p25_unit_price": round(p25, 2),
        "p75_unit_price": round(p75, 2),
        "estimate_eur_be_pvm": round(med * qty, 2),
        "range_eur_be_pvm": [round(p25 * qty, 2), round(p75 * qty, 2)],
        "used_analogs": len(ups),
    }

# -------------------- CORS --------------------
ALLOWED_ORIGINS = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "*").split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if ALLOWED_ORIGINS == ["*"] else ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------- Schemas --------------------
class Msg(BaseModel):
    role: str
    content: str

class EstimateRequest(BaseModel):
    text: str
    address: Optional[str] = ""
    history: Optional[List[Msg]] = None

# -------------------- Endpoints --------------------
@app.get("/health")
def health():
    return {"ok": True}

@app.post("/estimate")
def estimate(req: EstimateRequest):
    text = (req.text or "").strip()
    address = (req.address or "").strip()

    # 1) Parseris (OpenAI)
    parsed = parse_text(text)  # jei vėliau norėsi: parse_text(text, history, address)

    work_type = (parsed.get("work_type") or "OTHER").strip()
    qty = parsed.get("qty")
    unit = parsed.get("unit")
    needs = bool(parsed.get("needs_clarification"))
    questions = parsed.get("questions") or parsed.get("clarifying_questions") or []

    # 2) Jei modelis grąžino OTHER – bandome deterministic fallback (ypač siūlėms)
    if work_type == "OTHER":
        wt2 = fallback_work_type(text)
        if wt2 != "OTHER":
            work_type = wt2
            needs = False  # nuimam, jei vien dėl work_type trūko

    # 3) Jei trūksta kiekio ar vieneto – klausiame
    if (qty is None) or (unit is None):
        if not questions:
            t = norm(text)
            if "stov" in t:
                questions = ["Kiek aukštų keičiamas stovas? (pvz. 1, 2, 3 aukštai)"]
            elif ("siul" in t) or ("tarplokin" in t) or ("tarpblokin" in t):
                questions = ["Kiek metrų (m) siūlės sandarinama? (pvz. 25 m)"]
            elif "stog" in t:
                questions = ["Kiek m² remontuojama? (pvz. 12 m2)"]
            else:
                questions = ["Nurodykite kiekį ir vienetą (m, m2, vnt, aukštai)."]

        suggestions = []
        t = norm(text)
        if "stog" in t:
            suggestions = [
                {"work_type": "PITCHED_ROOF_TILE", "title": "Šlaitinis stogas (čerpės)"},
                {"work_type": "FLAT_ROOF", "title": "Plokščias stogas (bitumas/ruberoidas)"},
                {"work_type": "OTHER", "title": "Kitas (patikslinsiu)"},
            ]
        elif "stov" in t:
            suggestions = [
                {"work_type": "SEWER_STACK", "title": "Nuotekų stovas"},
                {"work_type": "WATER_STACK", "title": "Šalto/karšto vandens stovas"},
                {"work_type": "OTHER", "title": "Kitas (patikslinsiu)"},
            ]
        elif ("siul" in t) or ("tarplokin" in t) or ("tarpblokin" in t):
            suggestions = [
                {"work_type": "FACADE_SEAM", "title": "Tarplokinė siūlė fasade"},
                {"work_type": "PIPE", "title": "Vamzdyno remontas"},
                {"work_type": "OTHER", "title": "Kitas (patikslinsiu)"},
            ]

        return {
            "status": "need_more_info",
            "work_type_guess": work_type,
            "questions": questions,
            "suggestions": suggestions,
        }

    # 4) Net jei work_type liko OTHER, mes GALIM skaičiuoti pagal RAG analogus
    query = text + (" " + address if address else "")
    analogs = search_similar(query, limit=12)

    # 5) Skaičiuojam kainą
    try:
        qty_f = float(qty)
    except:
        return {
            "status": "need_more_info",
            "work_type_guess": work_type,
            "questions": ["Koks kiekis? (pvz. 25 m, 12 m2, 2 aukštai)"],
        }

    unit_s = str(unit).strip().lower()
    price = calc_from_analogs(analogs, qty_f, unit_s)

    if not price:
        return {
            "status": "no_price_model",
            "work_type_guess": work_type,
            "message": "Neradau pakankamai tinkamų analogų su tuo pačiu vienetu. Reikia patikslinti arba prašyti rangovo.",
            "analogs": analogs[:5],
        }

    return {
        "status": "ok",
        "work_type": work_type,
        "qty": qty_f,
        "unit": unit_s,
        "estimate_eur_be_pvm": price["estimate_eur_be_pvm"],
        "range_eur_be_pvm": price["range_eur_be_pvm"],
        "analogs": analogs[:5],
        "meta": {
            "unit_price_median": price["median_unit_price"],
            "unit_price_p25": price["p25_unit_price"],
            "unit_price_p75": price["p75_unit_price"],
            "used_analogs": price["used_analogs"],
        },
    }
