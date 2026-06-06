import re
import sqlite3
from sentence_transformers import SentenceTransformer
import sqlite_vec
from tqdm import tqdm

text = open(".local_data/books/gosudar.txt", encoding="utf-8").read()

parts = re.split(r"(?m)^(ГЛАВА\s+[IVXLCDM]+)\s*$", text)
chapters = []

for i in range(1, len(parts), 2):
    header = parts[i].strip()
    section = parts[i + 1].strip().splitlines()

    title = section[0].strip()
    body = "\n".join(section[1:]).strip()
    chapter_text = title + "\n\n" + body

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n+", body) if p.strip()]
    chapters.append((title, chapter_text, paragraphs))

model = SentenceTransformer("deepvk/USER2-base")
db = sqlite3.connect("book.sqlite")
db.enable_load_extension(True)
sqlite_vec.load(db)
db.enable_load_extension(False)

db.execute("""
CREATE TABLE IF NOT EXISTS chapters (
    id INTEGER PRIMARY KEY,
    title TEXT,
    text TEXT,
    embedding BLOB
)
""")

db.execute("""
CREATE TABLE IF NOT EXISTS paragraphs (
    id INTEGER PRIMARY KEY,
    chapter_id INTEGER,
    text TEXT,
    embedding BLOB
)
""")

for title, chapter_text, paragraphs in tqdm(chapters, desc="embedding chapters", total=len(chapters)):
    emb = model.encode(chapter_text, prompt_name="search_document")
    cur = db.execute(
        "INSERT INTO chapters (title, text, embedding) VALUES (?, ?, ?)",
        (title, chapter_text, sqlite_vec.serialize_float32(emb.tolist()))
    )
    chapter_id = cur.lastrowid

    for paragraph in tqdm(paragraphs, desc="embedding paragraphs", total=len(paragraphs)):
        p_emb = model.encode(paragraph, prompt_name="search_document")
        db.execute(
            "INSERT INTO paragraphs (chapter_id, text, embedding) VALUES (?, ?, ?)",
            (chapter_id, paragraph, sqlite_vec.serialize_float32(p_emb.tolist()))
        )

db.commit()
db.close()