import os
import requests
from db import get_conn

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")

def embed(text: str):
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY nenustatytas (Render Environment).")

    r = requests.post(
        "https://api.openai.com/v1/embeddings",
        headers={"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"},
        json={"model": EMBED_MODEL, "input": text[:8000]},
        timeout=60,
    )
    r.raise_for_status()
    return r.json()["data"][0]["embedding"]

def search_similar(query: str, limit: int = 12):
    vec = embed(query)
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            select registration_nr, address, title, qty, unit, cost, contractor
            from jobs
            order by embedding <-> %s
            limit %s
            """,
            (vec, limit),
        )
        rows = cur.fetchall()

    return [
        {
            "registration_nr": r[0],  # <-- pranešimo nr (KA pasitikrinimui)
            "address": r[1],
            "title": r[2],
            "qty": r[3],
            "unit": r[4],
            "cost_be_pvm": r[5],
            "contractor": r[6],
        }
        for r in rows
    ]
