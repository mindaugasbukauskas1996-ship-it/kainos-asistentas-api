import os
import psycopg
from openai import OpenAI

DB_URL = os.getenv("SUPABASE_DB_URL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=OPENAI_API_KEY)

def embed(text: str) -> list[float]:
    # naudok tą patį embedding modelį, kurį naudojai ingest'e
    r = client.embeddings.create(
        model=os.getenv("EMBED_MODEL", "text-embedding-3-small"),
        input=text
    )
    return r.data[0].embedding

def vec_to_pgvector(vec: list[float]) -> str:
    # pgvector tekstinis formatas: [0.1,0.2,0.3]
    return "[" + ",".join(f"{x:.6f}" for x in vec) + "]"

def search_similar(query: str, limit: int = 12):
    vec = embed(query)
    vec_str = vec_to_pgvector(vec)

    with psycopg.connect(DB_URL, sslmode="require") as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                select
                    job_id,
                    registration_nr,
                    title,
                    qty,
                    unit,
                    cost_be_pvm as cost_be_pvm,
                    contractor,
                    address,
                    (embedding <-> (%s)::vector) as distance
                from jobs
                order by embedding <-> (%s)::vector
                limit %s;
                """,
                (vec_str, vec_str, limit),
            )
            rows = cur.fetchall()
            cols = [d.name for d in cur.description]

    # grąžinam kaip dict'us
    out = []
    for r in rows:
        d = dict(zip(cols, r))
        out.append(d)
    return out
