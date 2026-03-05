import os
import psycopg
from openai import OpenAI


DB_URL = os.getenv("SUPABASE_DB_URL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")


def _must_env(name: str) -> str:
    v = os.getenv(name)
    if not v:
        raise RuntimeError(f"Missing env var: {name}")
    return v


def _client() -> OpenAI:
    key = _must_env("OPENAI_API_KEY")
    return OpenAI(api_key=key)


def embed(text: str):
    """
    Sugeneruoja embedding vektorių per OpenAI.
    """
    c = _client()
    r = c.embeddings.create(model=EMBED_MODEL, input=text)
    return r.data[0].embedding


def vec_to_pgvector(vec):
    """
    Konvertuoja Python list[float] į pgvector tekstinį formatą: [0.1,0.2,...]
    """
    # svarbu: be tarpų, su '.' kaip decimal separator
    return "[" + ",".join(f"{float(x)}" for x in vec) + "]"


def detect_domain(query: str) -> str:
    q = (query or "").lower()

    if ("tarplokin" in q) or ("tarpblokin" in q) or ("siūl" in q) or ("siul" in q) or ("sandarin" in q) or ("hermet" in q):
        return "seam"
    if ("stog" in q) or ("čerp" in q) or ("cerp" in q) or ("ruberoid" in q) or ("bitum" in q):
        return "roof"
    if ("stov" in q) or ("nuotek" in q) or ("vandens" in q) or ("karsto" in q) or ("salto" in q):
        return "stack"
    return "other"


def build_where_clause(domain: str) -> str:
    if domain == "seam":
        return """
        WHERE (
            lower(text_full) LIKE '%tarplokin%' OR
            lower(text_full) LIKE '%tarpblokin%' OR
            lower(text_full) LIKE '%siul%' OR
            lower(text_full) LIKE '%siūl%' OR
            lower(text_full) LIKE '%sandarin%' OR
            lower(text_full) LIKE '%hermet%' OR
            lower(text_full) LIKE '%mastik%' OR
            lower(text_full) LIKE '%poliuretan%'
        )
        """
    if domain == "roof":
        return """
        WHERE (
            lower(text_full) LIKE '%stog%' OR
            lower(text_full) LIKE '%cerp%' OR
            lower(text_full) LIKE '%čerp%' OR
            lower(text_full) LIKE '%ruberoid%' OR
            lower(text_full) LIKE '%bitum%' OR
            lower(text_full) LIKE '%pratek%' OR
            lower(text_full) LIKE '%skardin%'
        )
        """
    if domain == "stack":
        return """
        WHERE (
            lower(text_full) LIKE '%stov%' OR
            lower(text_full) LIKE '%nuotek%' OR
            lower(text_full) LIKE '%vandens%' OR
            lower(text_full) LIKE '%karsto%' OR
            lower(text_full) LIKE '%salto%' OR
            lower(text_full) LIKE '%trišak%' OR
            lower(text_full) LIKE '%trisak%'
        )
        """
    return ""


def search_similar(query: str, limit: int = 12):
    """
    Randa panašius įrašus iš Supabase (pgvector).
    """
    # 1) env patikra
    db_url = _must_env("SUPABASE_DB_URL")

    # 2) embedding
    vec = embed(query)
    vec_pg = vec_to_pgvector(vec)

    # 3) domeno filtras
    domain = detect_domain(query)
    where_sql = build_where_clause(domain)

    sql = f"""
    SELECT
        id,
        registration_nr,
        address,
        title,
        qty,
        unit,
        cost,
        contractor,
        text_full,
        (embedding <-> (%s)::vector) AS distance
    FROM jobs
    {where_sql}
    ORDER BY embedding <-> (%s)::vector
    LIMIT %s;
    """

    try:
        with psycopg.connect(db_url, sslmode="require") as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (vec_pg, vec_pg, limit))
                rows = cur.fetchall()
                cols = [d.name for d in cur.description]
        return [dict(zip(cols, r)) for r in rows]

    except Exception as e:
        # šitas erroras pasimatys Render loguose
        raise RuntimeError(f"rag_search failed: {type(e).__name__}: {e}")
