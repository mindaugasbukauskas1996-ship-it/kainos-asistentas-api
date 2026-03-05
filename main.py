from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import os
import re
import statistics
import unicodedata

from rag_search import search_similar
from openai_parser import parse_text  # tavo esamas parseris (turi turėti parse_text(text) -> dict)


app = FastAPI(title="Kainos asistentas API (RAG)")

# ---------------- CORS ----------------
ALLOWED_ORIGINS = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "*").split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if ALLOWED_ORIGINS == ["*"] else ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- Models ----------------
class Msg(BaseModel):
    role: str
    content: str

class EstimateRequest(BaseModel):
    text: str
    address: Optional[str] = ""
    history: Optional[List[Msg]] = None


# ---------------- Utils ----------------
def norm(s: str) -> str:
    s = (s or "").lower()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))  # ąčęėįšųūž → aceeisuuz
    return s

def pct(xs, p: float):
    if not xs:
        return None
    xs = sorted(xs)
    k = (len(xs) - 1) * p
    f = int(k)
    c = min(f + 1, len(xs) - 1)
    if f == c:
        return xs[f]
    return xs[f] + (xs[c] - xs[f]) * (k - f)

def extract_qty_unit_fallback(text: str):
    """
    Atsarginis kiekio/vieneto ištraukimas regex'u, jei parseris nepateikė.
    Grąžina (qty, unit) arba (None, None)
    """
    t = norm(text)

    # m2 (leidžiam: m2, m²)
    m2 = re.search(r"(\d+(?:[.,]\d+)?)\s*(m2|m²)\b", t)
    if m2:
        qty = float(m2.group(1).replace(",", "."))
        return qty, "m2"

    # metrai
    m = re.search(r"(\d+(?:[.,]\d+)?)\s*(m)\b", t)
    if m:
        qty = float(m.group(1).replace(",", "."))
        return qty, "m"

    # aukštai / aukštas
    aukst = re.search(
        r"(\d+(?:[.,]\d+)?)\s*(aukstu|aukstų|aukstų|aukstus|aukstai|aukštai|aukste|aukšte|aukstas|aukštas)\b",
        t
    )
    if aukst:
        qty = float(aukst.group(1).replace(",", "."))
        return qty, "aukštai"

    # vnt
    vnt = re.search(r"(\d+(?:[.,]\d+)?)\s*(vnt)\b", t)
    if vnt:
        qty = float(vnt.group(1).replace(",", "."))
        return qty, "vnt"

    return None, None

def guess_unit_from_keywords(text: str):
    """
    Jei vienetas neaiškus – bandome pagal raktinius žodžius.
    """
    t = norm(text)
    if "tarplokin" in t or "tarpblokin" in t or "siul" in t:
        return "m"
    if "stog" in t:
        return "m2"
    if "stov" in t:
        return "aukštai"
    return None

def make_suggestions(text: str):
    t = norm(text)
    if "stog" in t:
        return [
            {"title": "Plokščias stogas (bitumas/ruberoidas)", "work_hint": "STOGAS_PLOKSCIAS"},
            {"title": "Šlaitinis stogas (čerpės)", "work_hint": "STOGAS_CERPES"},
            {"title": "Kitas stogo darbas", "work_hint": "STOGAS_KITA"},
        ]
    if "stov" in t:
        return [
            {"title": "Nuotekų stovas", "work_hint": "STOVAS_NUOTEKU"},
            {"title": "Šalto/karšto vandens stovas", "work_hint": "STOVAS_VANDENS"},
            {"title": "Kitas stovo darbas", "work_hint": "STOVAS_KITA"},
        ]
    if "tarplokin" in t or "tarpblokin" in t or "siul" in t:
        return [
            {"title": "Tarplokinės siūlės sandarinimas fasade", "work_hint": "FASADAS_SIULE"},
            {"title": "Fasado remontas / defektų šalinimas", "work_hint": "FASADAS_REMONTAS"},
            {"title": "Kita (patikslinsiu)", "work_hint": "KITA"},
        ]
    return []

def calc_from_analogs(analogs: List[Dict[str, Any]], qty: float, unit: str):
    """
    Skaičiuojam medianinę vieneto kainą iš analogų:
    unit_price = cost / qty (tik jei analogas turi qty>0 ir unit sutampa)

    SVARBU: analoguose kaina gali ateiti kaip 'cost' (DB) arba 'cost_be_pvm' (jei kažkur alias'intas).
    Todėl imam: cost OR cost_be_pvm
    """
    ups = []
    unit_norm = (unit or "").strip().lower()

    for a in analogs:
        aq = a.get("qty")
        ac = a.get("cost")
        if ac is None:
            ac = a.get("cost_be_pvm")  # fallback, jei analoguose toks laukas
        au = (a.get("unit") or "").strip().lower()

        if aq is None or ac is None:
            continue
        if au != unit_norm:
            continue

        try:
            aqf = float(aq)
            acf = float(ac)
            if aqf > 0:
                ups.append(acf / aqf)
        except:
            continue

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
        "used_analogs": len(ups),
    }


# ---------------- Routes ----------------
@app.get("/health")
def health():
    return {"ok": True}

@app.post("/estimate")
def estimate(req: EstimateRequest):
    text = (req.text or "").strip()
    address = (req.address or "").strip()

    # 1) Parseris (OpenAI). Jei nulūžta – fallback.
    parsed: Dict[str, Any] = {}
    try:
        parsed = parse_text(text) or {}
    except Exception:
        parsed = {}

    qty = parsed.get("qty")
    unit = parsed.get("unit")
    work_type = (parsed.get("work_type") or "OTHER")

    # 2) Fallback: jei parseris nedavė qty/unit – regex
    if qty is None or unit is None:
        q2, u2 = extract_qty_unit_fallback(text)
        if qty is None:
            qty = q2
        if unit is None:
            unit = u2

    # 3) Jei vienetas vis dar None – spėjam iš raktinių žodžių
    if unit is None:
        unit = guess_unit_from_keywords(text)

    # 4) Jei trūksta kiekio arba vieneto – klausiame
    if qty is None or unit is None:
        questions = []
        t = norm(text)

        if "stov" in t:
            questions.append("Kiek aukštų keičiamas stovas? (pvz. 1, 2, 3 aukštai)")
        elif "tarplokin" in t or "tarpblokin" in t or "siul" in t:
            questions.append("Kiek metrų (m) siūlės sandarinama? (pvz. 25 m)")
        elif "stog" in t:
            questions.append("Kiek m² remontuojama? (pvz. 12 m2)")
        else:
            questions.append("Nurodykite kiekį ir vienetą (m, m2, vnt, aukštai).")

        return {
            "status": "need_more_info",
            "work_type_guess": work_type,
            "questions": questions,
            "suggestions": make_suggestions(text),
        }

    # 5) Retrieval: analogai iš Supabase pgvector
    query = text + ((" " + address) if address else "")
    analogs = search_similar(query, limit=12)

    # 6) Skaičiavimas
    try:
        qty_f = float(qty)
    except:
        return {
            "status": "need_more_info",
            "work_type_guess": work_type,
            "questions": ["Koks kiekis? (pvz. 25 m, 12 m2, 2 aukštai)"],
            "suggestions": make_suggestions(text),
        }

    price = calc_from_analogs(analogs, qty_f, str(unit))

    # 7) Jei nepavyko – grąžinam analogus (su registration_nr)
    if not price:
        return {
            "status": "no_price_model",
            "work_type_guess": work_type,
            "message": "Neradau pakankamai tinkamų analogų su tuo pačiu vienetu kainai patikimai įvertinti. Reikia patikslinti arba prašyti rangovo pasiūlymo.",
            "qty": qty_f,
            "unit": unit,
            "analogs": analogs[:12],
        }

    # 8) OK
    return {
        "status": "ok",
        "work_type": work_type,
        "qty": qty_f,
        "unit": unit,
        "estimate_eur_be_pvm": price["estimate_eur_be_pvm"],
        "range_eur_be_pvm": price["range_eur_be_pvm"],
        "analogs": analogs[:12],
        "meta": {
            "unit_price_median": price["median_unit_price"],
            "unit_price_p25": price["p25_unit_price"],
            "unit_price_p75": price["p75_unit_price"],
            "used_analogs": price["used_analogs"],
        },
    }
