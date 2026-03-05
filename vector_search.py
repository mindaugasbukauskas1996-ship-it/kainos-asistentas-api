import requests
import os
from db import get_conn

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def embed(text):

    r = requests.post(
        "https://api.openai.com/v1/embeddings",
        headers={
            "Authorization": f"Bearer {OPENAI_API_KEY}"
        },
        json={
            "model": "text-embedding-3-small",
            "input": text
        }
    )

    return r.json()["data"][0]["embedding"]


def search_similar(text):

    vector = embed(text)

    with get_conn() as conn:
        cur = conn.cursor()

        cur.execute("""
        select
            registration_nr,
            title,
            cost,
            qty,
            unit,
            contractor
        from jobs
        order by embedding <-> %s
        limit 20
        """, (vector,))

        rows = cur.fetchall()

    return rows
