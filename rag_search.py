import os
import psycopg
from openai import OpenAI

DB_URL = os.getenv("SUPABASE_DB_URL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=OPENAI_API_KEY)

def embed(text: str):
    r = client.embeddings.create(
        model=os.getenv("EMBED_MODEL", "text-embedding-3-small"),
        input=text
    )
    return r.data[0].embedding


def vec_to_pgvector(vec):
    # konvertuoja python list -> pgvector string
    return "[" + ",".join(str(x) for x in vec) + "]"


def search_similar(query: str, limit: int = 12):

    vec = embed(query)
    vec_pg = vec_to_pgvector(vec)

    with psycopg.connect(DB_URL, sslmode="require") as conn:
        with conn.cursor() as cur:

            cur.execute(
                """
                SELECT
                    job_id,
                    registration_nr_full as registration_nr,
                    samatos_pavadinimas as title,
                    qty_extracted as qty,
                    unit,
                    cost_be_pvm_eur as cost_be_pvm,
                    contractor,
                    address_final as address
                FROM jobs
                ORDER BY embedding <-> (%s)::vector
                LIMIT %s
                """,
                (vec_pg, limit),
            )

            rows = cur.fetchall()
            cols = [d.name for d in cur.description]

    result = []

    for r in rows:
        result.append(dict(zip(cols, r)))

    return result
