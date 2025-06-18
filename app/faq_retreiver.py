import pandas as pd
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

faq_data = pd.read_csv("data/faq.csv")
questions = faq_data["question"].tolist()
answers = faq_data["answer"].tolist()

# BM25
tokenized = [q.lower().split() for q in questions]
bm25 = BM25Okapi(tokenized)

# FAISS
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(questions)
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings))

def retrieve_faq_answer(query):
    bm25_results = bm25.get_top_n(query.lower().split(), questions, n=3)
    query_vec = model.encode([query])
    D, I = index.search(np.array(query_vec), k=3)
    faiss_results = [questions[i] for i in I[0]]
    combined = bm25_results + faiss_results
    return "\n".join([faq_data[faq_data.question == q]["answer"].values[0] for q in combined if q in faq_data.question.values])
