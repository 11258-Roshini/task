import pytesseract
from pdf2image import convert_from_path
import re
import faiss
import json
import numpy as np
import logging
from sentence_transformers import SentenceTransformer
import ollama
import sys

#  CONFIG 
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
PDF_PATH =r"C:\Users\Roshini.T\Downloads\1001015828.pdf"
POPPLER_PATH = r"C:\poppler-22.12.0\Library\bin"

EMBED_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "llama3:8b"
CHUNK_SIZE = 250 
TOP_K = 2
LOG_FILE = "rag_extraction.log"

#  LOGGING 
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("RAG")

#  OCR 
def extract_text_ocr(pdf_path):
    text = ""
    pages = convert_from_path(pdf_path, poppler_path=POPPLER_PATH)
    for i, img in enumerate(pages, 1):
        page_text = pytesseract.image_to_string(img)
        text += f"\n=== PAGE {i} ===\n{page_text}"
    return text

#  SAFE RULE HELPERS 
def looks_like_name(text):
    if not text:
        return False
    if len(text.split()) > 4:
        return False
    if re.search(r"\d|admit|drink|patient|history|diagnosis", text, re.I):
        return False
    return bool(re.fullmatch(r"[A-Za-z .'-]{3,40}", text))

def extract_patient_name(text):
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    for line in lines[:60]:
        m = re.search(r"\bR\s*e\s*[:\-]?\s*(.+)", line, re.I)
        if m:
            candidate = m.group(1).strip()
            if looks_like_name(candidate):
                return candidate
    return ""

def extract_dob(text):
    m = re.search(
        r"(DOB|Date of Birth)\s*[:\-]?\s*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})",
        text,
        re.I
    )
    return m.group(2) if m else ""

def extract_order_id(text):
    m = re.search(r"Order ID\s*:\s*([A-Z0-9\-()]+)", text, re.I)
    return m.group(1) if m else ""

#  SPLIT PATIENT SECTIONS 
def split_patient_sections(text):
    pattern = re.compile(r"(?=(?:Patient Name\s*:|Order ID\s*:|Re\s*:))", re.I)
    points = [m.start() for m in pattern.finditer(text)]
    sections = []
    for i in range(len(points)):
        start = points[i]
        end = points[i + 1] if i + 1 < len(points) else len(text)
        section = text[start:end].strip()
        if len(section) > 80:
            sections.append(section)
    return sections

#  FACILITY (RULE ONLY) 
def extract_facility(section):
    keywords = ["clinic", "hospital", "medical center", "health center"]
    for line in section.splitlines()[:40]:
        for kw in keywords:
            if kw.lower() in line.lower():
                return line.strip()
    return ""

#  RAG 
def rag_extract(section, embed_model):
    words = section.split()
    chunks = [" ".join(words[i:i + CHUNK_SIZE]) for i in range(0, len(words), CHUNK_SIZE)]
    embeddings = embed_model.encode(chunks)

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings).astype("float32"))

    qvec = embed_model.encode(["physician name"]).astype("float32")
    _, idx = index.search(qvec, TOP_K)
    context = "\n".join(chunks[i] for i in idx[0])

    prompt = f"""
Extract physician name only.
Return JSON only.

{{
  "physician_first": "",
  "physician_last": ""
}}

Context:
{context}
"""

    response = ollama.chat(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0}
    )

    match = re.search(r"\{[\s\S]*\}", response["message"]["content"])
    if not match:
        return {"physician_first": "", "physician_last": ""}

    try:
        data = json.loads(match.group())
    except:
        return {"physician_first": "", "physician_last": ""}

    # Validate against OCR text
    for k in data:
        if data[k] and data[k].lower() not in section.lower():
            data[k] = ""

    return data

#  MAIN 
def main():
    text = extract_text_ocr(PDF_PATH)
    sections = split_patient_sections(text)
    embed_model = SentenceTransformer(EMBED_MODEL)

    results = []

    for section in sections:
        patient_name = extract_patient_name(section)
        dob = extract_dob(section)
        order_id = extract_order_id(section)
        facility = extract_facility(section)

        rag_data = rag_extract(section, embed_model)

        first, last = ("", "")
        if patient_name:
            parts = patient_name.split()
            first = parts[0]
            last = parts[-1] if len(parts) > 1 else ""

        results.append({
            "patient": {
                "first_name": first,
                "last_name": last,
                "dob": dob,
                "order_id": order_id
            },
            "physician": {
                "first_name": rag_data["physician_first"],
                "last_name": rag_data["physician_last"],
                "facility_name": facility
            }
        })

    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    main()
