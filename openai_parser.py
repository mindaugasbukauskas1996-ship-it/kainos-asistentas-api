import os
import requests
import json

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")

SYSTEM_PROMPT = """
Tu esi daugiabučių remonto darbų klasifikatorius.

Grąžink tik JSON.

Galimi work_type:

SEWER_STACK
WATER_STACK
FACADE_SEAM
FLAT_ROOF
PITCHED_ROOF_TILE
PIPE
OTHER

Taisyklės:

- stovas visada matuojamas aukštais
- trišakis galioja tik nuotekų stovui
- tarplokinės siūlės matuojamos metrais
- čerpės gali būti vnt arba m2
- jei kiekio nėra -> needs_clarification true

JSON struktūra:

{
 "work_type": "...",
 "qty": number | null,
 "unit": "...",
 "needs_clarification": true/false,
 "questions": []
}
"""


def parse_text(text):

    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": MODEL,
        "input": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text}
        ]
    }

    r = requests.post(
        "https://api.openai.com/v1/responses",
        headers=headers,
        json=payload
    )

    data = r.json()

    try:
        content = data["output"][0]["content"][0]["text"]
        return json.loads(content)
    except:
        return {
            "work_type": "OTHER",
            "qty": None,
            "unit": None,
            "needs_clarification": True,
            "questions": ["Nepavyko atpažinti darbo tipo"]
        }
