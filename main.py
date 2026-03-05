from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import os
import re
import statistics

from rag_search import search_similar, detect_domain

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
    """
    t = (text or "").lower()

    m2 = re.search(r"(\d+(?:[.,]\d+)?)\s*(m2|m²)\b", t)
    if m2:
        return float(m2.group(1).replace(",", ".")), "m2"

    m = re.search(r"(\d+(?:[.,]\d+)?)\s*m\b", t)
    if m:
        return float(m.group(1).replace(",", ".")), "m"

    a = re.search(r"(\d+(?:[.,]\d+)?)\s*(aukst|aukšt)\b", t)
    if a:
        return float(a.group(1).replace(",", ".")), "aukstas"

    v = re.search(r"(\d+(?:[.,]\d+)?)\s*(vnt|v)\b", t)
    if v:
        return float(v.group(1).replace(",", ".")), "vnt"

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
    if domain == "seam":
        return "m"
    if domain == "roof":
        return "m2"
    if domain == "stack":
        return "aukstas"
    return ""


def calc_from_analogs(analogs: List[Dict[str, Any]], qty: float, unit: str):
    """
    Skaičiuoja iš analogų:
    unit_price = cost / qty
    Naudoja tik analogus su qty ir cost > 0.
    Jei analogų unit tuščias – leidžiam (nes jūsų duomenyse taip dažnai būna).
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


def build_chat_reply_ok(work_type: str, qty: float, unit: str, estimate: float, rng: List[float], analogs: List[Dict[str, Any]]):
    lines = []
    lines.append(f"Preliminari kaina (be PVM): **{estimate:.2f} €**")
    lines.append(f"Intervalas (be PVM): **{rng[0]:.2f}–{rng[1]:.2f} €**")
    lines.append(f"Kiekis: **{qty:g} {unit}**")
    lines.append("")
    lines.append("Panašūs analogai (patikrai pagal pranešimo nr.):")
    for a in analogs[:5]:
        reg = a.get("registration_nr") or "-"
        title = a.get("title") or ""
        addr = a.get("address") or ""
        cost = a.get("cost")
        u = a.get("unit") or ""
        q = a.get("qty")
        if cost is None or q is None:
            lines.append(f"• **{reg}** — {title} ({addr})")
        else:
            lines.append(f"• **{reg}** — {title} ({addr}) | {q} {u} | {float(cost):.2f} €")
    return "\n".join(lines)


def build_chat_reply_need_more(domain: str):
    if domain == "seam":
        return (
            "Kad paskaičiuočiau, reikia kiekio.\n"
            "Parašyk, **kiek metrų (m)** tarplokinės siūlės sandarinama (pvz. „25 m“)."
        )
    if domain == "roof":
        return (
            "Kad paskaičiuočiau, reikia kiekio.\n"
            "Parašyk, **kiek m²** stogo remontuojama (pvz. „12 m2“)."
        )
    if domain == "stack":
        return (
            "Kad paskaičiuočiau, reikia kiekio.\n"
            "Parašyk, **kiek aukštų** keičiamas stovas (pvz. „2 aukštai“)."
        )
    return "Kad paskaičiuočiau, nurodyk kiekį (pvz. 25 m / 12 m2 / 2 aukštai / 3 vnt)."


def suggestions_for_domain(domain: str):
    if domain == "seam":
        return [
            {"title": "Tarplokinė siūlė fasade", "example": "Reikia sandarinti 25 m tarplokinės siūlės fasade"},
            {"title": "Sandarinti siūles (pelėsis bute)", "example": "Reikia sandarinti 12 m siūlių fasade, dėl pelėsio bute"},
        ]
    if domain == "roof":
        return [
            {"title": "Plokščias stogas", "example": "Reikia sutvarkyti 12 m2 plokščio stogo dangos"},
            {"title": "Šlaitinis stogas (čerpės)", "example": "Reikia pakeisti 10 m2 čerpių (šlaitinis stogas)"},
        ]
    if domain == "stack":
        return [
            {"title": "Stovas (aukštai)", "example": "Reikia pakeisti 2 aukštų stovą"},
            {"title": "Stovas su trišakiu (nuotekoms)", "example": "Reikia pakeisti 2 aukštų nuotekų stovą su trišakiu"},
        ]
    return []


@app.post("/estimate")
def estimate(req: EstimateRequest):
    text = (req.text or "").strip()
    address = (req.address or "").strip()

    query = text + (" " + address if address else "")

    # Domenas -> work_type + default unit
    domain = detect_domain(query)
    work_type = work_type_from_domain(domain)
    default_unit = unit_from_domain(domain)

    # Kiekis iš teksto
    qty, unit_guess = extract_qty(text)

    # Jei nėra kiekio – grąžinam chat klausimą + suggestions
    if qty is None:
        return {
            "status": "need_more_info",
            "work_type_guess": work_type,
            "reply_text": build_chat_reply_need_more(domain),
            "questions": ["Nurodykite kiekį (su vienetu)."],
            "suggestions": suggestions_for_domain(domain),
        }

    unit = unit_guess or default_unit
    if domain == "seam":
        unit = "m"  # priverstinai
    if domain == "roof":
        unit = "m2"
    if domain == "stack":
        unit = "aukstas"

    # RAG analogai
    analogs = search_similar(query, limit=12)

    # Skaičiavimas
    price = calc_from_analogs(analogs, float(qty), unit)
    if not price:
        return {
            "status": "no_price_model",
            "work_type_guess": work_type,
            "reply_text": "Neradau pakankamai tinkamų analogų su qty/cost šiam darbui patikimai įvertinti. Reikia patikslinti arba prašyti rangovo pasiūlymo.",
            "analogs": analogs[:5],
        }

    top5 = price["used_analogs"][:5]
    reply = build_chat_reply_ok(
        work_type=work_type,
        qty=float(qty),
        unit=unit,
        estimate=price["estimate"],
        rng=price["range"],
        analogs=top5,
    )

    return {
        "status": "ok",
        "work_type": work_type,
        "qty": float(qty),
        "unit": unit,
        "estimate_eur_be_pvm": price["estimate"],
        "range_eur_be_pvm": price["range"],
        "reply_text": reply,
        "suggestions": suggestions_for_domain(domain),
        "analogs": top5,
        "meta": {
            "domain": domain,
            "unit_price_median": price["unit_price_median"],
            "unit_price_p25": price["unit_price_p25"],
            "unit_price_p75": price["unit_price_p75"],
            "used_analogs": len(price["used_analogs"]),
        }
    }
