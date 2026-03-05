import os
import psycopg
from dotenv import load_dotenv

load_dotenv()

DB_URL = os.getenv("SUPABASE_DB_URL")

def get_conn():
    if not DB_URL:
        raise RuntimeError("SUPABASE_DB_URL nenustatytas (Render Environment).")
    return psycopg.connect(DB_URL, sslmode="require")
