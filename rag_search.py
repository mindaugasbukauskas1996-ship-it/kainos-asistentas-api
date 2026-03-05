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
        headers={
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json",
        },
        json={"model": EMBED_MODEL, "input": text[:8000]},
        timeout=60,
    )
    if r.status_code != 200:
        raise RuntimeError(f"OpenAI embeddings error {r.status_code}: {r.text}")
    j = r.json()
    return j["data"][0]["embedding"]

def search_similar(user_text: str, limit: int = 12):
    vec = embed(user_text)
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
        analogs.append({
            "registration_nr": r[0],
            "address": r[1],
            "title": r[2],
            "qty": r[3],
            "unit": r[4],
            "cost_be_pvm": r[5],
            "contractor": r[6],
        })
    return analogs
