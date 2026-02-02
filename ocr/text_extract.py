# import pdfplumber
# import pytesseract
# from PIL import Image
# import subprocess
# import json
# import re
# from collections import defaultdict

# # -------------------------------------------------
# # CONFIG
# # -------------------------------------------------
# TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
# OLLAMA_PATH = r"C:\Users\Roshini.T\AppData\Local\Programs\Ollama\ollama.exe"
# MODEL_NAME = "llama3"

# pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH


# # -------------------------------------------------
# # OCR FALLBACK
# # -------------------------------------------------
# def ocr_page(page):
#     try:
#         img = page.to_image(resolution=300).original
#         return pytesseract.image_to_string(img)
#     except Exception as e:
#         return ""


# # -------------------------------------------------
# # PDF → TEXT
# # -------------------------------------------------
# def pdf_to_text(pdf_path):
#     text = ""
#     with pdfplumber.open(pdf_path) as pdf:
#         for i, page in enumerate(pdf.pages, start=1):
#             extracted = page.extract_text()
#             if not extracted or len(extracted.strip()) < 10:
#                 extracted = ocr_page(page)
#             text += f"\n\n===== PAGE {i} =====\n{extracted}"
#     return text


# # -------------------------------------------------
# # CHUNKING
# # -------------------------------------------------
# def chunk_text(text, size=1200, overlap=200):
#     chunks = []
#     start = 0
#     while start < len(text):
#         end = min(start + size, len(text))
#         chunks.append(text[start:end])
#         start += size - overlap
#     return chunks


# # -------------------------------------------------
# # PROMPT-BASED EXTRACTION (STRICT)
# # -------------------------------------------------
# def ollama_extract(chunk):
#     prompt = f"""
# You are a medical document field extractor.

# DOCUMENT STRUCTURE:
# 1. Facility/Physician Information
#    - Physician First Name
#    - Physician Last Name

# 2. Patient Information
#    - First Name
#    - Last Name

# 3. DOB
# 4. Order ID
# 5. Facility Name

# RULES:
# - Extract ONLY if value appears directly near its label
# - DO NOT infer or guess
# - Ignore medical history, progress notes, vitals
# - Ignore DOBs in narrative text
# - Prefer FIRST valid occurrence
# - If missing, return empty string
# - Verify each value is label-associated

# Return STRICT JSON ONLY.

# JSON KEYS:
# physician_first_name
# physician_last_name
# patient_first_name
# patient_last_name
# dob
# order_id
# facility_name

# TEXT:
# {chunk}
# """

#     try:
#         proc = subprocess.run(
#             [OLLAMA_PATH, "run", MODEL_NAME],
#             input=prompt,
#             text=True,
#             encoding="utf-8",        # FIX UnicodeDecodeError
#             errors="ignore",
#             capture_output=True
#         )

#         output = proc.stdout.strip()
#         match = re.search(r"\{.*\}", output, re.DOTALL)
#         return json.loads(match.group()) if match else {}

#     except Exception:
#         return {}


# # -------------------------------------------------
# # REGEX VALIDATION (HYBRID SAFETY)
# # -------------------------------------------------
# def validate_with_regex(field, value):
#     patterns = {
#         "dob": r"\b\d{2}[/-]\d{2}[/-]\d{4}\b",
#         "order_id": r"[A-Z]{2,}\d+",
#     }

#     if field in patterns and value:
#         return bool(re.search(patterns[field], value))
#     return True


# # -------------------------------------------------
# # CONFIDENCE SCORING
# # -------------------------------------------------
# def calculate_confidence(entry):
#     score = 0
#     for field, value in entry.items():
#         if value:
#             score += 1
#             if validate_with_regex(field, value):
#                 score += 1
#     return score


# # -------------------------------------------------
# # MULTI-CHUNK RECONCILIATION
# # -------------------------------------------------
# def reconcile_results(results):
#     field_votes = defaultdict(list)

#     for res in results:
#         for field, value in res.items():
#             if value:
#                 field_votes[field].append(value)

#     final = {}
#     audit = {}

#     for field, values in field_votes.items():
#         chosen = max(set(values), key=values.count)
#         final[field] = chosen
#         audit[field] = {
#             "chosen": chosen,
#             "occurrences": values.count(chosen)
#         }

#     return final, audit


# # -------------------------------------------------
# # MAIN EXTRACTION
# # -------------------------------------------------
# def extract_pdf_fields(pdf_path):
#     text = pdf_to_text(pdf_path)
#     chunks = chunk_text(text)

#     candidates = []

#     for chunk in chunks:
#         entry = ollama_extract(chunk)
#         if entry and any(entry.values()):
#             candidates.append(entry)

#     if not candidates:
#         return {}, {}

#     final_data, audit_log = reconcile_results(candidates)

#     return final_data, audit_log


# # -------------------------------------------------
# # RUN
# # -------------------------------------------------
# if __name__ == "__main__":
#     pdf_file = r"C:\Users\Roshini.T\Downloads\merge_sample_1.pdf"

#     data, audit = extract_pdf_fields(pdf_file)

#     print("\n===== FINAL RESULT =====")
#     print(json.dumps(data, indent=4))

#     print("\n===== AUDIT LOG =====")
#     print(json.dumps(audit, indent=4))




# -------------- new prompt code ----------------------


import pdfplumber
import pytesseract
import subprocess
import json
import re

# ----------------------------
# Configure Tesseract path
# ----------------------------
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

OLLAMA_PATH = r"C:\Users\Roshini.T\AppData\Local\Programs\Ollama\ollama.exe"
MODEL_NAME = "llama3"

# ----------------------------
# OCR helper
# ----------------------------
def ocr_page(page):
    try:
        img = page.to_image(resolution=300).original
        return pytesseract.image_to_string(img)
    except Exception as e:
        print(f"OCR failed: {e}")
        return ""

# ----------------------------
# PDF → Text
# ----------------------------
def pdf_to_text(pdf_path):
    full_text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page_no, page in enumerate(pdf.pages, start=1):
            text = page.extract_text()
            if not text or len(text.strip()) < 20:
                text = ocr_page(page)

            full_text += f"\n\n===== PAGE {page_no} =====\n{text}"

    return full_text

# ----------------------------
# Split document by PATIENT
# ----------------------------
def split_by_patient(text):
    """
    Split document into patient sections using multiple real-world patterns
    """
    pattern = r"(Patient Information|Patient Name|PATIENT DETAILS|PATIENT\s*:)"
    
    parts = re.split(pattern, text, flags=re.IGNORECASE)

    patients = []
    buffer = ""

    for part in parts:
        if re.match(pattern, part, flags=re.IGNORECASE):
            if buffer.strip():
                patients.append(buffer)
            buffer = part
        else:
            buffer += part

    if buffer.strip():
        patients.append(buffer)

    # If still only one block, treat entire document as one patient
    if len(patients) == 1:
        return patients

    return patients


# ----------------------------
# Ollama extraction (CLI)
# ----------------------------

def ollama_extract(patient_text):
    prompt = f"""
    You are a highly accurate medical document information extraction engine.

    TASK:
    Extract structured data ONLY from the text provided below.
    Do NOT guess, infer, or assume any information.
    If a field is not explicitly present, return an empty string.

    IMPORTANT RULES:
    - Extract information ONLY from labels and nearby text
    - Do NOT merge multiple patients
    - Do NOT repeat values across patients
    - Do NOT include extra text or explanations
    - Return STRICTLY valid JSON
    - If multiple values exist, choose the FIRST clearly labeled value

    FIELDS TO EXTRACT:
    physician_first_name
    physician_last_name
    patient_first_name
    patient_last_name
    dob
    order_id
    facility_name

    LABEL GUIDANCE:
    - Physician First Name appears near "Physician First Name:"
    - Physician Last Name appears near "Physician Last Name:"
    - Patient First Name appears near "Patient Information" or "First Name:"
    - Patient Last Name appears near "Last Name:"
    - DOB appears near "DOB:" or "Date of Birth:"
    - Order ID appears near "Order ID:" or "Order Number:"
    - Facility Name appears near "Facility Name:" or "Facility/Physician Information:"

    FORMAT RULES:
    - Dates must be in MM/DD/YYYY format
    - Names must not contain addresses, phone numbers, or titles (Dr., MD)
    - Trim whitespace and special characters
    - Use empty string "" if value is missing

    OUTPUT FORMAT (JSON ONLY):
    {{
    "physician_first_name": "",
    "physician_last_name": "",
    "patient_first_name": "",
    "patient_last_name": "",
    "dob": "",
    "order_id": "",
    "facility_name": ""
    }}

    TEXT:
    \"\"\"
    {patient_text}
    \"\"\"
    """





    try:
        result = subprocess.run(
            [OLLAMA_PATH, "run", MODEL_NAME],
            input=prompt,
            text=True,
            encoding="utf-8",
            errors="ignore",
            capture_output=True
        )

        output = result.stdout.strip()

        match = re.search(r"\{.*\}", output, re.DOTALL)
        if match:
            return json.loads(match.group())

    except Exception as e:
        print(f"Ollama failed: {e}")

    return {}

# ----------------------------
# MAIN PIPELINE
# ----------------------------
def extract_pdf_fields(pdf_path):
    text = pdf_to_text(pdf_path)

    patient_sections = split_by_patient(text)
    print(f"Detected patient sections: {len(patient_sections)}")
    # 🚑 fallback: no patient split detected
    if not patient_sections:
        patient_sections = [text]

    results = []
    seen_patients = set()

    for patient_text in patient_sections:
        entry = ollama_extract(patient_text)

        if entry and any(entry.values()):
            key = (
                entry.get("patient_first_name", "").lower(),
                entry.get("patient_last_name", "").lower(),
                entry.get("dob", "")
            )

            if key not in seen_patients:
                seen_patients.add(key)
                results.append(entry)

    return results


# ----------------------------
# RUN
# ----------------------------
if __name__ == "__main__":
    pdf_file = r"C:\Users\Roshini.T\Downloads\merge_sample_1.pdf"

    final_result = extract_pdf_fields(pdf_file)

    print("\n✅ FINAL OUTPUT\n")
    print(json.dumps(final_result, indent=4, ensure_ascii=False))





