from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import os
import re
import statistics

from rag_search import search_similar, detect_domain  # naudosim tavo rag_search domenui


app = FastAPI(title="Kainos asistentas API (RAG)")

ALLOWED_ORIGINS = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "*").split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if ALLOWED_ORIGINS == ["*"] else ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"ok": True}


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
    k = (len(xs) - 1) * p
    f = int(k)
    c = min(f + 1, len(xs) - 1)
    if f == c:
        return xs[f]
    return xs[f] + (xs[c] - xs[f]) * (k - f)


def extract_qty(text: str):
    """
    Ištraukia kiekį:
      - "25 m", "12 m2", "2 aukšt", "2 aukst", "2 aukštų"
    Grąžina (qty: float|None, unit_guess: str|None)
    """
    t = (text or "").lower()

    # m2 / m²
    m2 = re.search(r"(\d+(?:[.,]\d+)?)\s*(m2|m²)\b", t)
    if m2:
        qty = float(m2.group(1).replace(",", "."))
        return qty, "m2"

    # metrai (m)
    m = re.search(r"(\d+(?:[.,]\d+)?)\s*m\b", t)
    if m:
        qty = float(m.group(1).replace(",", "."))
        return qty, "m"

    # aukštai
    a = re.search(r"(\d+(?:[.,]\d+)?)\s*(aukst|aukšt)\b", t)
    if a:
        qty = float(a.group(1).replace(",", "."))
        return qty, "aukstas"

    # vnt
    v = re.search(r"(\d+(?:[.,]\d+)?)\s*(vnt|v)\b", t)
    if v:
        qty = float(v.group(1).replace(",", "."))
        return qty, "vnt"

    return None, None


def work_type_from_domain(domain: str) -> str:
    if domain == "seam":
        return "FACADE_SEAM"
    if domain == "roof":
        return "ROOF"
    if domain == "stack":
        return "STACK"
    return "OTHER"


def unit_from_domain(domain: str) -> str:
    # pagal tavo taisykles:
    # siūlė = m
    # stogas = m2
    # stovas = aukštas
    if domain == "seam":
        return "m"
    if domain == "roof":
        return "m2"
    if domain == "stack":
        return "aukstas"
    return ""


def calc_from_analogs(analogs: List[Dict[str, Any]], qty: float, unit: str):
    """
    Skaičiuoja kainą iš analogų:
    unit_price = cost / qty
    Naudojam tik analogus, kurie turi qty ir cost > 0.
    Jei analogų unit tuščias – nefiltruojam per unit, o remiamės iškviestu 'unit' iš domeno.
    """
    ups = []
    used = []

    for a in analogs:
        aq = a.get("qty")
        ac = a.get("cost")
        if aq is None or ac is None:
            continue
        try:
            aq = float(aq)
            ac = float(ac)
        except:
            continue
        if aq <= 0 or ac <= 0:
            continue

        # jei analogas turi unit ir jis nesutampa – praleidžiam
        au = (a.get("unit") or "").lower().strip()
        if au and unit and au != unit.lower():
            continue

        ups.append(ac / aq)
        used.append(a)

    if not ups:
        return None

    med = statistics.median(ups)
    p25 = pct(ups, 0.25)
    p75 = pct(ups, 0.75)

    return {
        "estimate": round(med * qty, 2),
        "range": [round(p25 * qty, 2), round(p75 * qty, 2)],
        "unit_price_median": round(med, 2),
        "unit_price_p25": round(p25, 2),
        "unit_price_p75": round(p75, 2),
        "used_analogs": used,
    }


@app.post("/estimate")
def estimate(req: EstimateRequest):
    text = (req.text or "").strip()
    address = (req.address or "").strip()

    # 1) Nustatom domeną + work_type + default unit
    domain = detect_domain(text + (" " + address if address else ""))
    work_type = work_type_from_domain(domain)
    default_unit = unit_from_domain(domain)

    # 2) Ištraukiam qty + unit iš teksto
    qty, unit_guess = extract_qty(text)

    # 3) Jei nėra qty – klausiam
    if qty is None:
        if domain == "seam":
            return {
                "status": "need_more_info",
                "work_type_guess": work_type,
                "questions": ["Kiek metrų (m) tarplokinės siūlės sandarinama? (pvz. 25 m)"],
                "suggestions": [
                    {"title": "Tarplokinė siūlė fasade (m)", "example": "Reikia sandarinti 25 m tarplokinės siūlės fasade"}
                ]
            }
        if domain == "roof":
            return {
                "status": "need_more_info",
                "work_type_guess": work_type,
                "questions": ["Kiek m² stogo remontuojama? (pvz. 12 m2)"],
                "suggestions": [
                    {"title": "Plokščias stogas (m2)", "example": "Reikia sutvarkyti 12 m2 plokščio stogo dangos"},
                    {"title": "Šlaitinis stogas (m2)", "example": "Reikia pakeisti 10 m2 čerpių (šlaitinis stogas)"},
                ]
            }
        if domain == "stack":
            return {
                "status": "need_more_info",
                "work_type_guess": work_type,
                "questions": ["Kiek aukštų keičiamas stovas? (pvz. 1 aukštas, 2 aukštai)"],
                "suggestions": [
                    {"title": "Stovas (aukštai)", "example": "Reikia pakeisti 2 aukštų stovą"}
                ]
            }

        return {
            "status": "need_more_info",
            "work_type_guess": work_type,
            "questions": ["Nurodykite kiekį (pvz. 25 m / 12 m2 / 2 aukštai / 3 vnt)."]
        }

    # 4) Unit: jei tekste parašyta unit – imam ją, kitaip imam domeno default
    unit = unit_guess or default_unit

    # jei domenas seam – priverstinai m (kad neklaidintų)
    if domain == "seam":
        unit = "m"

    # 5) RAG analogai
    query = text + (" " + address if address else "")
    analogs = search_similar(query, limit=12)

    # 6) Skaičiuojam
    price = calc_from_analogs(analogs, float(qty), unit)
    if not price:
        return {
            "status": "no_price_model",
            "work_type_guess": work_type,
            "message": "Neradau pakankamai tinkamų analogų su qty/cost šiam darbui patikimai įvertinti. Reikia patikslinti arba prašyti rangovo pasiūlymo.",
            "analogs": analogs[:5],
        }

    # 7) Grąžinam top 5 analogus su registration_nr
    top5 = price["used_analogs"][:5]
    return {
        "status": "ok",
        "work_type": work_type,
        "qty": float(qty),
        "unit": unit,
        "estimate_eur_be_pvm": price["estimate"],
        "range_eur_be_pvm": price["range"],
        "analogs": top5,
        "meta": {
            "domain": domain,
            "unit_price_median": price["unit_price_median"],
            "unit_price_p25": price["unit_price_p25"],
            "unit_price_p75": price["unit_price_p75"],
            "used_analogs": len(price["used_analogs"]),
        }
    }
