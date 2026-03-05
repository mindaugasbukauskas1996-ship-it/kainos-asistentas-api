import os
import psycopg

DB_URL = os.getenv("SUPABASE_DB_URL")

def get_conn():
    return psycopg.connect(DB_URL)
