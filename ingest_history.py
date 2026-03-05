import csv
import requests
from db import get_conn
import os

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def embed(text):

    r = requests.post(
        "https://api.openai.com/v1/embeddings",
        headers={"Authorization": f"Bearer {OPENAI_API_KEY}},
        json={"model":"text-embedding-3-small","input":text}
    )

    return r.json()["data"][0]["embedding"]


with open("historical_jobs_min.csv") as f:

    reader = csv.DictReader(f)

    with get_conn() as conn:
        cur = conn.cursor()

        for r in reader:

            text = r["samatos_pavadinimas"]

            vector = embed(text)

            cur.execute("""
            insert into jobs
            (registration_nr,address,title,qty,unit,cost,contractor,text_full,embedding)
            values (%s,%s,%s,%s,%s,%s,%s,%s,%s)
            """,(
                r["registration_nr_full"],
                r["address_final"],
                r["samatos_pavadinimas"],
                r["qty_extracted"],
                r["unit"],
                r["cost_be_pvm_eur"],
                r["contractor"],
                text,
                vector
            ))

        conn.commit()
