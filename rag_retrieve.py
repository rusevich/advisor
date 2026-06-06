import sqlite3
import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("deepvk/USER2-base")
db = sqlite3.connect("book.sqlite")

def cosine_similarity(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))

def blob_to_vec(blob):
    return np.frombuffer(blob, dtype=np.float32)

def top_k(items, query_vec, k):
    scored = [(cosine_similarity(query_vec, vec), item) for item, vec in items]
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:k]

def retrieve_5_most_relevant(question):
    qvec = model.encode(question, prompt_name="search_query")

    chapters = db.execute(
        "SELECT id, title, text, embedding FROM chapters"
    ).fetchall()
    chapter_items = [
        ({"id": cid, "title": title, "text": text}, blob_to_vec(emb))
        for cid, title, text, emb in chapters
    ]
    best_chapters = top_k(chapter_items, qvec, 3)

    paragraph_items = []
    for _, chapter in best_chapters:
        paragraphs = db.execute(
            "SELECT id, text, embedding FROM paragraphs WHERE chapter_id = ?",
            (chapter["id"],)
        ).fetchall()
        paragraph_items += [
            ({"id": pid, "text": text, "chapter_title": chapter.get("title", "")}, blob_to_vec(emb))
            for pid, text, emb in paragraphs
        ]

    best_paragraphs = top_k(paragraph_items, qvec, 5)

    result = [(content.get("chapter_title", ""), content.get("text", "")) for _, content in best_paragraphs ]

    return result

if __name__ == "__main__":
    while True:
        question = input("Что тебе сказать? ")
        res = rag(question)
        for title, text in res:
            print(title)
            print(text)