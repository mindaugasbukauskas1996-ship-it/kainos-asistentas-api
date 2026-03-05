from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import os
import re
import statistics

from openai_parser import parse_text   # tavo esamas parseris
from rag_search import search_similar  # naujas

app = FastAPI(title="Kainos asistentas API (RAG)")

ALLOWED_ORIGINS = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "*").split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if ALLOWED_ORIGINS == ["*"] else ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Msg(BaseModel):
    role: str
    content: str

class EstimateRequest(BaseModel):
    text: str
    address: Optional[str] = ""
    history: Optional[List[Msg]] = None

def pct(xs, p):
    if not xs:
        return None
    xs = sorted(xs)
    k = (len(xs)-1) * p
    f = int(k)
    c = min(f+1, len(xs)-1)
    if f == c:
        return xs[f]
    return xs[f] + (xs[c]-xs[f])*(k-f)

def calc_from_analogs(analogs: List[Dict[str, Any]], qty: float, unit: str):
    # naudosim tik analogus su unit_price arba (cost/qty) jei yra qty
    ups = []
    for a in analogs:
        aq = a.get("qty")
        ac = a.get("cost_be_pvm")
        au = (a.get("unit") or "").lower()
        if aq and ac and au == (unit or "").lower():
            try:
                aq = float(aq)
                ac = float(ac)
                if aq > 0:
                    ups.append(ac / aq)
            except:
                pass

    if not ups:
        return None

    med = statistics.median(ups)
    p25 = pct(ups, 0.25)
    p75 = pct(ups, 0.75)

    return {
        "median_unit_price": round(med, 2),
        "p25_unit_price": round(p25, 2),
        "p75_unit_price": round(p75, 2),
        "estimate_eur_be_pvm": round(med * qty, 2),
        "range_eur_be_pvm": [round(p25 * qty, 2), round(p75 * qty, 2)],
        "used_analogs": len(ups)
    }

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/estimate")
def estimate(req: EstimateRequest):
    text = (req.text or "").strip()
    address = (req.address or "").strip()

    # 1) parseris: work_type, qty, unit, questions
    hist = [m.model_dump() for m in (req.history or [])]
    parsed = parse_text(text)  # jei tavo parse_text priima history/address – galėsim išplėsti vėliau

    work_type = (parsed.get("work_type") or "OTHER")
    qty = parsed.get("qty")
    unit = parsed.get("unit")

    needs = bool(parsed.get("needs_clarification"))
    questions = parsed.get("questions") or parsed.get("clarifying_questions") or []

    # 2) Jei trūksta kiekio – klausiame kaip chate
    if needs or (qty is None) or (unit is None):
        # jei parseris nepateikė, pridedam loginius klausimus
        if not questions:
            # stovas -> aukštai
            if "stov" in text.lower():
                questions = ["Kiek aukštų keičiamas stovas? (pvz. 1, 2, 3 aukštai)"]
            elif "siūl" in text.lower() or "tarplokin" in text.lower():
                questions = ["Kiek metrų (m) siūlės sandarinama? (pvz. 25 m)"]
            elif "stog" in text.lower():
                questions = ["Kiek m² remontuojama? (pvz. 12 m2)"]
            else:
                questions = ["Nurodykite kiekį ir vienetą (m, m2, vnt, aukštai)."]

        # pasiūlymai (3 variantai) – pagal raktinius žodžius
        suggestions = []
        t = text.lower()
        if "stog" in t:
            suggestions = [
                {"work_type":"PITCHED_ROOF_TILE","title":"Šlaitinis stogas (čerpės)"},
                {"work_type":"FLAT_ROOF","title":"Plokščias stogas (bitumas/ruberoidas)"},
                {"work_type":"OTHER","title":"Kitas (patikslinsiu)"},
            ]
        elif "stov" in t:
            suggestions = [
                {"work_type":"SEWER_STACK","title":"Nuotekų stovas"},
                {"work_type":"WATER_STACK","title":"Šalto/karšto vandens stovas"},
                {"work_type":"OTHER","title":"Kitas (patikslinsiu)"},
            ]
        elif "siūl" in t or "tarplokin" in t:
            suggestions = [
                {"work_type":"FACADE_SEAM","title":"Tarplokinė siūlė fasade"},
                {"work_type":"PIPE","title":"Vamzdyno remontas"},
                {"work_type":"OTHER","title":"Kitas (patikslinsiu)"},
            ]

        return {
            "status": "need_more_info",
            "work_type_guess": work_type,
            "questions": questions,
            "suggestions": suggestions,
        }

    # 3) Retrieval: randam analogus iš Supabase (visada grąžins registration_nr)
    analogs = search_similar(text + (" " + address if address else ""), limit=12)

    # 4) Skaičiuojam kainą iš analogų (unit price median + p25/p75)
    try:
        qty_f = float(qty)
    except:
        return {
            "status": "need_more_info",
            "work_type_guess": work_type,
            "questions": ["Koks kiekis? (pvz. 25 m, 12 m2, 2 aukštai)"],
        }

    price = calc_from_analogs(analogs, qty_f, str(unit))

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
        "unit": unit,
        "estimate_eur_be_pvm": price["estimate_eur_be_pvm"],
        "range_eur_be_pvm": price["range_eur_be_pvm"],
        "analogs": analogs[:5],
        "meta": {
            "unit_price_median": price["median_unit_price"],
            "unit_price_p25": price["p25_unit_price"],
            "unit_price_p75": price["p75_unit_price"],
            "used_analogs": price["used_analogs"],
        }
    }
