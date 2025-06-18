from fastapi import FastAPI, Request
from app.graph import process_query

app = FastAPI()

@app.post("/process-query")
async def handle_query(request: Request):
    data = await request.json()
    email = data.get("email")
    query = data.get("query")
    return process_query(email, query)


