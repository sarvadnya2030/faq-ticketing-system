import requests
import os
from dotenv import load_dotenv

load_dotenv()

TOKEN = os.getenv("HUBSPOT_API_TOKEN")
BASE = os.getenv("HUBSPOT_BASE_URL")
HEADERS = {
    "Authorization": f"Bearer {TOKEN}",
    "Content-Type": "application/json"
}

def get_contact_info(email):
    url = f"{BASE}/crm/v3/objects/contacts?email={email}"
    res = requests.get(url, headers=HEADERS)
    return res.json()

def create_ticket(email, query, response):
    payload = {
        "properties": {
            "subject": f"Escalated: {query[:50]}",
            "content": f"User asked: {query}\nAI responded: {response}",
            "hs_pipeline": "0",
            "hs_pipeline_stage": "1"
        }
    }
    url = f"{BASE}/crm/v3/objects/tickets"
    res = requests.post(url, json=payload, headers=HEADERS)
    return res.json().get("id", "unknown")
