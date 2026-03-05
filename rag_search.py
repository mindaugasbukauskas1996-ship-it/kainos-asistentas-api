import os
import psycopg
from openai import OpenAI

DB_URL = os.getenv("SUPABASE_DB_URL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=OPENAI_API_KEY)

EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")


def embed(text: str):
    r = client.embeddings.create(
        model=EMBED_MODEL,
        input=text
    )
    return r.data[0].embedding


def vec_to_pgvector(vec):
    # pgvector tekstinis formatas: [0.1,0.2,0.3]
    return "[" + ",".join(str(x) for x in vec) + "]"


def search_similar(query: str, limit: int = 12):
    vec = embed(query)
    vec_pg = vec_to_pgvector(vec)

    with psycopg.connect(DB_URL, sslmode="require") as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    id,
                    registration_nr,
                    address,
                    title,
                    qty,
                    unit,
                    cost as cost_be_pvm,
                    contractor,
                    text_full,
                    (embedding <-> (%s)::vector) as distance
                FROM jobs
                ORDER BY embedding <-> (%s)::vector
                LIMIT %s;
                """,
                (vec_pg, vec_pg, limit),
            )
            rows = cur.fetchall()
            cols = [d.name for d in cur.description]

    return [dict(zip(cols, r)) for r in rows]
