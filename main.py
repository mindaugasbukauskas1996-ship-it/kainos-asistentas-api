import os
import re
import math
import statistics
from typing import Optional, List, Dict, Any

import requests
import psycopg
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


# =========================
# Config
# =========================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini").strip()
OPENAI_EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small").strip()

SUPABASE_DB_URL = os.getenv("SUPABASE_DB_URL", "").strip()

ALLOWED_ORIGINS_ENV = os.getenv("ALLOWED_ORIGINS", "*").strip()
ALLOWED_ORIGINS = ["*"] if ALLOWED_ORIGINS_ENV == "*" else [x.strip() for x in ALLOWED_ORIGINS_ENV.split(",") if x.strip()]


# =========================
# App
# =========================
app = FastAPI(title="Kainos asistentas API (RAG + Chat)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================
# Schemas
# =========================
class Msg(BaseModel):
    role: str  # "user" or "assistant"
    content: str


class EstimateRequest(BaseModel):
    text: str
    address: Optional[str] = ""
    history: Optional[List[Msg]] = None


# =========================
# Helpers: DB
# =========================
def get_conn():
    if not SUPABASE_DB_URL:
        raise RuntimeError("SUPABASE_DB_URL nenustatytas (Render Environment).")
    return psycopg.connect(SUPABASE_DB_URL, sslmode="require")


# =========================
# Helpers: Embeddings + Retrieval
# =========================
def embed(text: str) -> List[float]:
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY nenustatytas (Render Environment).")

    r = requests.post(
        "https://api.openai.com/v1/embeddings",
        headers={
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": OPENAI_EMBED_MODEL,
            "input": (text or "")[:8000],
        },
        timeout=60,
    )
    if r.status_code != 200:
        raise RuntimeError(f"OpenAI embeddings error {r.status_code}: {r.text}")
    j = r.json()
    return j["data"][0]["embedding"]


def search_similar(query: str, limit: int = 12) -> List[Dict[str, Any]]:
    # Returns analogs with registration_nr (pranešimo nr)
    vec = embed(query)

    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            select
              registration_nr,
              address,
              title,
              qty,
              unit,
              cost,
              contractor
            from jobs
            order by embedding <-> %s
            limit %s
            """,
            (vec, limit),
        )
        rows = cur.fetchall()

    analogs = []
    for r in rows:
        analogs.append(
            {
                "registration_nr": r[0],
                "address": r[1],
                "title": r[2],
                "qty": r[3],
                "unit": r[4],
                "cost_be_pvm": r[5],
                "contractor": r[6],
            }
        )
    return analogs


# =========================
# Helpers: math
# =========================
def pct(xs: List[float], p: float) -> float:
    xs = sorted(xs)
    if not xs:
        return 0.0
    k = (len(xs) - 1) * p
    f = int(k)
    c = min(f + 1, len(xs) - 1)
    if f == c:
        return xs[f]
    return xs[f] + (xs[c] - xs[f]) * (k - f)


def calc_from_analogs(analogs: List[Dict[str, Any]], qty: float, unit: str) -> Optional[Dict[str, Any]]:
    # Unit price from analogs: cost/qty for matching unit
    ups: List[float] = []
    unit_norm = (unit or "").strip().lower()

    for a in analogs:
        au = (a.get("unit") or "").strip().lower()
        aq = a.get("qty")
        ac = a.get("cost_be_pvm")
        if au != unit_norm:
            continue
        if aq in (None, "", 0) or ac in (None, "", 0):
            continue
        try:
            aqf = float(aq)
            acf = float(ac)
            if aqf > 0:
                ups.append(acf / aqf)
        except:
            continue

    # jei analogų per mažai – kainos intervalas bus „iš oro“
    if len(ups) < 3:
        return None

    med = statistics.median(ups)
    p25 = pct(ups, 0.25)
    p75 = pct(ups, 0.75)

    return {
        "unit_price_median": round(med, 2),
        "unit_price_p25": round(p25, 2),
        "unit_price_p75": round(p75, 2),
        "estimate_eur_be_pvm": round(med * qty, 2),
        "range_eur_be_pvm": [round(p25 * qty, 2), round(p75 * qty, 2)],
        "used_analogs": len(ups),
    }


# =========================
# Fallback parser (if no OpenAI)
# =========================
UNIT_ALIASES = {
    "m²": "m2",
    "m2": "m2",
    "m": "m",
    "vnt": "vnt",
    "aukštas": "aukstas",
    "aukštai": "aukstas",
    "aukstu": "aukstas",
    "aukštų": "aukstas",
}


def extract_qty_unit(text: str):
    t = (text or "").lower().replace("m²", "m2")
    pats = [
        (r"(\d+(?:[.,]\d+)?)\s*m2\b", "m2"),
        (r"(\d+(?:[.,]\d+)?)\s*m\b", "m"),
        (r"(\d+(?:[.,]\d+)?)\s*vnt\b", "vnt"),
        (r"(\d+(?:[.,]\d+)?)\s*auk", "aukstas"),
    ]
    for rgx, u in pats:
        m = re.search(rgx, t)
        if m:
            val = float(m.group(1).replace(",", "."))
            return val, u
    return None, None


def quick_suggestions(text: str) -> List[Dict[str, str]]:
    t = (text or "").lower()
    if "stog" in t:
        return [
            {"work_type": "PITCHED_ROOF_TILE", "title": "Šlaitinis stogas (čerpės)"},
            {"work_type": "FLAT_ROOF", "title": "Plokščias stogas (bitumas/ruberoidas)"},
            {"work_type": "OTHER", "title": "Kita (patikslinsiu)"},
        ]
    if "stov" in t:
        return [
            {"work_type": "SEWER_STACK", "title": "Nuotekų stovas"},
            {"work_type": "WATER_STACK", "title": "Šalto/karšto vandens stovas"},
            {"work_type": "OTHER", "title": "Kita (patikslinsiu)"},
        ]
    if "tarplokin" in t or "tarpblok" in t or "siūl" in t or "siul" in t:
        return [
            {"work_type": "FACADE_SEAM", "title": "Tarplokinė siūlė fasade"},
            {"work_type": "PIPE", "title": "Vamzdynas (vanduo/nuotekos)"},
            {"work_type": "OTHER", "title": "Kita (patikslinsiu)"},
        ]
    return []


def fallback_work_type(text: str) -> str:
    t = (text or "").lower()
    if "tarplokin" in t or "tarpblok" in t or ("siūl" in t and "fasad" in t) or ("siul" in t and "fasad" in t):
        return "FACADE_SEAM"
    if "nuotek" in t or "kanaliz" in t:
        return "SEWER_STACK"
    if "stov" in t:
        return "WATER_STACK"  # default; realiai gali būti ir sewer, bet be AI geriau klausti
    if "stog" in t or "čerpi" in t or "cerpi" in t:
        return "PITCHED_ROOF_TILE"
    if "bitum" in t or "ruberoid" in t or "plokšč" in t or "ploks" in t:
        return "FLAT_ROOF"
    if "vamzd" in t:
        return "PIPE"
    return "OTHER"


def fallback_questions(work_type: str, qty: Optional[float], unit: Optional[str]) -> List[str]:
    qs = []
    if work_type in ("SEWER_STACK", "WATER_STACK"):
        # stovas – tik aukštais
        if qty is None or unit != "aukstas":
            qs.append("Kiek aukštų keičiamas stovas? (pvz. 1 aukštas, 2 aukštai, 3 aukštai)")
        if work_type == "SEWER_STACK":
            qs.append("Ar su trišakiu? (taip/ne)")
    elif work_type == "FACADE_SEAM":
        if qty is None or unit != "m":
            qs.append("Kiek metrų (m) tarplokinės siūlės sandarinama? (pvz. 25 m)")
    elif work_type in ("PITCHED_ROOF_TILE", "FLAT_ROOF"):
        if qty is None or unit != "m2":
            qs.append("Kiek m² stogo remontuojama? (pvz. 12 m2)")
    elif qty is None or unit is None:
        qs.append("Nurodykite kiekį su vienetu (m, m2, vnt arba aukštai).")
    return qs


# =========================
# OpenAI parser (structured JSON)
# =========================
SYSTEM_PROMPT = """
Tu esi daugiabučių remonto darbų kainos asistento parseris.
Tikslas: iš vartotojo teksto (ir pokalbio istorijos) nustatyti darbo tipą, kiekį ir vienetą, arba užduoti tikslinamuosius klausimus.

Svarbios taisyklės:
- "stovas" matuojamas TIK AUKŠTAIS (aukštai). Nenaudoti metrų stovui.
- "trišakis" galioja TIK NUOTEKŲ stovui.
- tarplokinės (tarpblokinės) siūlės fasade matuojamos metrais (m).
- stogo remontas dažniausiai m2; šlaitinis (čerpės) ir plokščias (bitumas/ruberoidas) yra skirtingi tipai.
- jei informacijos trūksta – grąžink needs_clarification=true ir klausimus.
- jei dviprasmiška – pateik suggestions (3 pasirinkimai).

Grąžink TIK JSON be jokio papildomo teksto.

Leistini work_type:
PITCHED_ROOF_TILE, FLAT_ROOF, FACADE_SEAM, SEWER_STACK, WATER_STACK, PIPE, OTHER

JSON formatas:
{
  "work_type": "...",
  "qty": number|null,
  "unit": "m"|"m2"|"vnt"|"aukstas"|null,
  "needs_clarification": true|false,
  "questions": [ ... ],
  "suggestions": [ {"work_type":"...","title":"..."} ]
}
"""


def parse_with_openai(text: str, history: Optional[List[Dict[str, str]]] = None) -> Optional[Dict[str, Any]]:
    if not OPENAI_API_KEY:
        return None

    # Suformuojam trumpą history (kad neišpūst konteksto)
    hist = history or []
    hist_trim = hist[-8:]  # paskutinės 8 žinutės

    user_blob = "Pokalbio istorija:\n"
    for m in hist_trim:
        role = m.get("role", "")
        content = (m.get("content", "") or "").strip()
        if not content:
            continue
        user_blob += f"- {role}: {content}\n"
    user_blob += "\nNauja vartotojo žinutė:\n" + (text or "")

    # Responses API
    r = requests.post(
        "https://api.openai.com/v1/responses",
        headers={
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": OPENAI_MODEL,
            "input": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_blob},
            ],
        },
        timeout=60,
    )

    if r.status_code != 200:
        # jei OpenAI trumpam nulūžta – nenumušam viso endpointo, tiesiog fallback
        return None

    j = r.json()
    try:
        content = j["output"][0]["content"][0]["text"]
        return requests.utils.json.loads(content)  # safe-ish json parse
    except Exception:
        # kartais modelis grąžina JSON, bet ne per tą raktą – fallback
        try:
            txt = j.get("output_text", "")
            if txt:
                return requests.utils.json.loads(txt)
        except Exception:
            return None
    return None


# =========================
# Endpoints
# =========================
@app.get("/health")
def health():
    return {"ok": True}


@app.post("/estimate")
def estimate(req: EstimateRequest):
    text = (req.text or "").strip()
    address = (req.address or "").strip()
    history = [m.model_dump() for m in (req.history or [])]

    # 1) Parser (OpenAI if possible, otherwise fallback)
    parsed = parse_with_openai(text, history=history)

    if not isinstance(parsed, dict):
        wt = fallback_work_type(text)
        qty, unit = extract_qty_unit(text)
        # jei stovas – priverstinis unit
        if wt in ("SEWER_STACK", "WATER_STACK"):
            unit = "aukstas" if unit == "aukstas" else unit
        parsed = {
            "work_type": wt,
            "qty": qty,
            "unit": unit,
            "needs_clarification": True if (qty is None or unit is None) else False,
            "questions": fallback_questions(wt, qty, unit),
            "suggestions": quick_suggestions(text),
        }

    work_type = (parsed.get("work_type") or "OTHER")
    qty = parsed.get("qty")
    unit = parsed.get("unit")
    needs = bool(parsed.get("needs_clarification"))
    questions = parsed.get("questions") or []
    suggestions = parsed.get("suggestions") or []

    # 2) Retrieval analogų (visada bandome, kad KA matytų pranešimo nr net ir prieš patikslinimą)
    analog_query = text + (" " + address if address else "")
    analogs: List[Dict[str, Any]] = []
    try:
        # Jei nėra OPENAI key – retrieval neišeis, nes embed() naudoja OpenAI.
        if OPENAI_API_KEY and SUPABASE_DB_URL:
            analogs = search_similar(analog_query, limit=12)
    except Exception:
        analogs = []

    # 3) Jei reikia patikslinimo – grąžinam klausimus + pasiūlymus + analogus (top 5)
    if needs or qty is None or unit is None:
        if not questions:
            questions = fallback_questions(work_type, qty, unit)
        if not suggestions:
            suggestions = quick_suggestions(text)

        return {
            "status": "need_more_info",
            "work_type_guess": work_type,
            "questions": questions,
            "suggestions": suggestions,
            "analogs": analogs[:5],
        }

    # 4) Skaičiuojam kainą iš analogų
    try:
        qty_f = float(qty)
    except Exception:
        return {
            "status": "need_more_info",
            "work_type_guess": work_type,
            "questions": ["Parašyk kiekį skaičiumi su vienetu (pvz. 25 m, 12 m2, 2 aukštai)."],
            "suggestions": suggestions or quick_suggestions(text),
            "analogs": analogs[:5],
        }

    price = calc_from_analogs(analogs, qty_f, str(unit))
    if not price:
        return {
            "status": "no_price_model",
            "work_type_guess": work_type,
            "message": "Neradau pakankamai tinkamų analogų su tuo pačiu vienetu kainai patikimai įvertinti. Reikia patikslinti arba prašyti rangovo pasiūlymo.",
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
            "unit_price_median": price["unit_price_median"],
            "unit_price_p25": price["unit_price_p25"],
            "unit_price_p75": price["unit_price_p75"],
            "used_analogs": price["used_analogs"],
        },
    }
