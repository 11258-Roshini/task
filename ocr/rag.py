

#-------------- below code faclity only wrong ------------------
import pytesseract
from pdf2image import convert_from_path
import re
import faiss
import json
import numpy as np
import logging
from sentence_transformers import SentenceTransformer
import ollama

# CONFIG
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
# PDF_PATH = r"C:\Users\Roshini.T\Downloads\merge_sample_1.pdf"
# PDF_PATH = r"C:\Users\Roshini.T\Downloads\sample_inbound_73.pdf"

POPPLER_PATH = r"C:\poppler-22.12.0\Library\bin"
EMBED_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "llama3:8b"
CHUNK_SIZE = 250
TOP_K = 2
LOG_FILE = "rag_extraction.log"

# LOGGING
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("RAG")

# OCR EXTRACTION
def extract_text_ocr(pdf_path):
    text = ""
    pages = convert_from_path(pdf_path, poppler_path=POPPLER_PATH)
    for page_num, page_image in enumerate(pages, start=1):
        page_text = pytesseract.image_to_string(page_image)
        text += f"\n=== PAGE {page_num} ===\n{page_text}"
    with open("ocr_output.txt", "w", encoding="utf-8") as f:
        f.write(text)
    return text

# REGEX HELPERS
def extract_order_id(text):
    m = re.search(
        r"Order ID\s*:\s*([A-Z0-9]+-\([A-Z0-9]+\))",
        text,
        re.I
    )
    return m.group(1).strip() if m else ""

def validate_order_id(order_id, section_text):
    if not order_id:
        return ""
    clean = re.sub(r"\s+", "", order_id)
    if clean.lower() in section_text.replace(" ", "").lower():
        return clean
    return ""

def validate_field(value, section_text):
    if not value:
        return ""
    return value if value.lower() in section_text.lower() else ""

# SPLIT MULTI-PATIENT
def split_patient_sections(text):
    pattern = re.compile(
        r"(?=(?:Patient Name\s*:|Order ID\s*:))",
        re.I
    )
    points = [m.start() for m in pattern.finditer(text)]
    sections = []
    for i in range(len(points)):
        start = points[i]
        end = points[i + 1] if i + 1 < len(points) else len(text)
        section = text[start:end].strip()
        if len(section) > 50:
            sections.append(section)
    logger.info(f"Detected patients: {len(sections)}")
    return sections

# SAFE JSON PARSE
def safe_json_loads(text, idx):
    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        logger.error(f"No JSON returned for patient {idx}")
        logger.error(text)
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError as e:
        logger.error(f"JSON parse error patient {idx}: {e}")
        return None

# FACILITY DETECTION
def detect_facility_name(section_text):
    keywords = ["clinic", "hospital", "doctor office", "medical center", "health center", "primary care"]
    for line in section_text.splitlines():
        line_clean = line.strip()
        for kw in keywords:
            if kw.lower() in line_clean.lower():
                return line_clean
    return ""

def clean_facility_name(text):
    keywords = ["clinic", "hospital", "doctor office", "medical center", "health center", "primary care"]
    parts = re.split(r"[-–,]| at ", text, flags=re.I)
    for part in parts:
        part_clean = part.strip()
        for kw in keywords:
            if kw.lower() in part_clean.lower():
                return part_clean
    return ""

# RAG EXTRACTION
def rag_extract_patient(section_text, embed_model, patient_index):

    order_id = extract_order_id(section_text)
    words = section_text.split()
    chunks = [" ".join(words[i:i + CHUNK_SIZE]) for i in range(0, len(words), CHUNK_SIZE)]

    embeddings = embed_model.encode(chunks)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings).astype("float32"))

    qvec = embed_model.encode(["Extract patient and physician details"]).astype("float32")
    _, idx = index.search(qvec, TOP_K)

    context = "\n".join(chunks[i] for i in idx[0])

    prompt = f"""
Extract details ONLY from context.
Return JSON only.

{{
  "patient": {{
    "first_name": "",
    "last_name": "",
    "dob": "",
    "order_id": ""
  }},
  "physician": {{
    "first_name": "",
    "last_name": "",
    "facility_name": ""
  }}
}}

Context:
{context}
"""

    response = ollama.chat(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0}
    )
    raw = response["message"]["content"]
    result = safe_json_loads(raw, patient_index)
    if not result:
        return None

    # ORDER ID
    order_id_clean = validate_order_id(order_id, section_text)
    result["patient"]["order_id"] = order_id_clean.upper() if order_id_clean else ""

    # FACILITY NAME CLEANING
    
    facility_raw = detect_facility_name(section_text)
    facility_name = clean_facility_name(facility_raw)

    # PHYSICIAN NAME FIX
    phys_full = result["physician"].get("first_name", "").strip()

    # If facility looks like doctor, merge into physician
    if facility_name.lower().startswith("dr.") or any(part in facility_name for part in phys_full.split()):
        phys_full = f"{phys_full} {facility_name}".strip() if phys_full else facility_name
        facility_name = ""  # clear facility

    # Split physician full name
    if phys_full:
        parts = phys_full.replace("Dr.", "").strip().split()
        if len(parts) >= 2:
            result["physician"]["first_name"] = f"Dr. {parts[0]}"
            result["physician"]["last_name"] = parts[-1]
        else:
            result["physician"]["first_name"] = f"Dr. {phys_full}"
            result["physician"]["last_name"] = ""
    else:
        result["physician"]["first_name"] = ""
        result["physician"]["last_name"] = ""

    # Set cleaned facility
    result["physician"]["facility_name"] = facility_name

    # VALIDATE LLM FIELDS
    result["patient"]["first_name"] = validate_field(result["patient"].get("first_name", ""), section_text)
    result["patient"]["last_name"] = validate_field(result["patient"].get("last_name", ""), section_text)
    result["patient"]["dob"] = validate_field(result["patient"].get("dob", ""), section_text)
    result["physician"]["first_name"] = validate_field(result["physician"].get("first_name", ""), section_text)
    result["physician"]["last_name"] = validate_field(result["physician"].get("last_name", ""), section_text)

    # LOG ONLY FINAL JSON
    try:
        final_json_str = json.dumps(result, indent=2)
        logger.info(f"Patient {patient_index} JSON:\n{final_json_str}")
    except Exception as e:
        logger.error(f"Error logging JSON for patient {patient_index}: {e}")

    return result

# MAIN
def main():
    logger.info("===== RAG EXTRACTION STARTED =====")
    text = extract_text_ocr(PDF_PATH)
    sections = split_patient_sections(text)

    embed_model = SentenceTransformer(EMBED_MODEL)
    seen = set()
    results = []

    for i, section in enumerate(sections, start=1):
        data = rag_extract_patient(section, embed_model, i)
        if not data:
            continue

        key = (
            data["patient"]["first_name"],
            data["patient"]["last_name"],
            data["patient"]["dob"],
            data["patient"]["order_id"]
        )

        if key in seen:
            logger.info(f"Duplicate skipped: {key}")
            continue

        seen.add(key)
        results.append(data)

    print(json.dumps(results, indent=2))
    logger.info("===== RAG EXTRACTION FINISHED =====")

if __name__ == "__main__":
    main()

#***************************************************************************************************************#

#👍 below code retrieves all correct name  in cc, re: in image :

import pytesseract
from PIL import Image
import re
import json
import logging

# CONFIG
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
IMAGE_PATH = r"C:\Users\Roshini.T\Downloads\image (2).png"
LOG_FILE = "patient_extraction.log"

# LOGGING
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("OCR")

# OCR

def extract_text_from_image(image_path):
    img = Image.open(image_path)
    text = pytesseract.image_to_string(img)
    logger.info("OCR extraction completed")
    return text

# NAME VALIDATOR
def looks_like_name(text):
    if not text:
        return False

    text = text.strip()

    if len(text) > 40:
        return False

    parts = text.split()
    if len(parts) < 2 or len(parts) > 4:
        return False

    bad_words = {
        "that", "with", "without", "more", "less", "than",
        "admits", "drinks", "thinks", "should", "also",
        "because", "history", "patient", "reports", "office"
    }

    lower = text.lower()
    if any(w in lower for w in bad_words):
        return False

    return True

# DOB NEAR NAME
def extract_dob_for_patient(text, patient_name):
    if not patient_name:
        return ""

    idx = text.lower().find(patient_name.lower())
    if idx == -1:
        return ""

    window = text[idx: idx + 300]

    patterns = [
        r"\((\d{2}/\d{2}/\d{4})\)",
        r"Date\s*of\s*Birth\s*[:\-]\s*(\d{1,2}/\d{1,2}/\d{4})",
        r"DOB\s*[:\-]\s*(\d{1,2}/\d{1,2}/\d{4})",
        r"\b\d{1,2}/\d{1,2}/\d{4}\b",
        r"\b[A-Za-z]{3}\s+\d{1,2},\s+\d{4}\b",
    ]

    for p in patterns:
        m = re.search(p, window, re.I)
        if m:
            return m.group(1)

    return ""

# PATIENT EXTRACTION (FINAL)
def extract_patient(text):
    lines = [l.strip() for l in text.splitlines() if l.strip()]

    # 🔑 Scan header region only
    for line in lines[:60]:

        # Accept: Re, Re:, Re., R e :
        m = re.search(
            r"\bR\s*e\s*[:.\-]?\s*([A-Za-z ,.'\-]+)\s*\(?(\d{2}/\d{2}/\d{4})?\)?",
            line,
            re.I
        )

        if m:
            name = (m.group(1) or "").strip()
            dob = (m.group(2) or "").strip()

            if looks_like_name(name):
                if not dob:
                    dob = extract_dob_for_patient(text, name)
                logger.info("Patient extracted from Re line")
                return name, dob

    # ---- FALLBACKS ----
    fallback_patterns = [
        r"Reg\s*[:\-]\s*([A-Za-z ,.'\-]+)",
        r"CC\s*Name\s*[:\-]\s*([A-Za-z ,.'\-]+)",
        r"Patient\s*[:\-]\s*([A-Za-z ,.'\-]+)",
    ]

    for pat in fallback_patterns:
        m = re.search(pat, text, re.I)
        if m:
            name = m.group(1).strip()
            if looks_like_name(name):
                dob = extract_dob_for_patient(text, name)
                logger.info("Patient extracted from fallback")
                return name, dob

    logger.warning("Patient not found")
    return "", ""

# MAIN
def main():
    logger.info("===== PATIENT EXTRACTION STARTED =====")

    text = extract_text_from_image(IMAGE_PATH)

    patient_name, dob = extract_patient(text)

    result = {
        "patient": {
            "name": patient_name,
            "dob": dob
        }
    }

    print(json.dumps(result, indent=2))
    logger.info("Final JSON generated")
    logger.info("===== PATIENT EXTRACTION FINISHED =====")

if __name__ == "__main__":
    main()
