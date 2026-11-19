"""
Multimodal Medical Report Image & PDF Analyzer for CardioPulse XAI v2.0.
Supports:
  1. Online Cloud Vision (Gemini Vision REST API via GEMINI_API_KEY) when online_mode=True
  2. Local Offline OCR (pytesseract for images, pdfplumber for digital PDFs) + Pillow preprocessing
  3. Clinical Regex & Entity Extraction Engine mapping raw report text to the 13 UCI features
     plus patient history flags (medications, smoking, diabetes, hypertension).
"""
from __future__ import annotations
import base64
import io
import json
import os
import re
import urllib.request
import urllib.error
from pathlib import Path
from PIL import Image, ImageEnhance, ImageFilter, ImageOps, ImageDraw, ImageFont


def preprocess_report_image(image_bytes: bytes) -> tuple[Image.Image, dict]:
    """Enhance scanned/photographed medical report for OCR and inspect quality."""
    img = Image.open(io.BytesIO(image_bytes))
    orig_size = img.size
    fmt = img.format or "PNG"
    gray = ImageOps.grayscale(img)
    enhanced = ImageEnhance.Contrast(gray).enhance(1.8)
    sharpened = enhanced.filter(ImageFilter.SHARPEN)
    quality = {
        "width": orig_size[0],
        "height": orig_size[1],
        "format": fmt,
        "resolution_ok": orig_size[0] >= 400 and orig_size[1] >= 300,
    }
    return sharpened, quality


def extract_clinical_entities(text: str) -> dict:
    """Parse clinical biomarkers and history cues from raw lab/ECG report text."""
    t = text or ""
    low = t.lower()
    extracted = {}
    confidence = {}

    def find_num(patterns, cast=float, lo=None, hi=None):
        for pat in patterns:
            m = re.search(pat, t, flags=re.IGNORECASE)
            if m:
                try:
                    val = cast(m.group(1))
                    if (lo is None or val >= lo) and (hi is None or val <= hi):
                        return val
                except ValueError:
                    continue
        return None

    age = find_num([r"\bage\s*[:=\-]?\s*(\d{2})\b", r"\b(\d{2})\s*(?:y/o|yrs|years old)\b"], int, 20, 95)
    if age is not None:
        extracted["age"] = age; confidence["age"] = "high"

    if re.search(r"\b(male|sex\s*[:=]?\s*m\b|gender\s*[:=]?\s*m\b)", low) and not re.search(r"\bfemale\b", low):
        extracted["sex"] = 1; confidence["sex"] = "high"
    elif re.search(r"\b(female|sex\s*[:=]?\s*f\b|gender\s*[:=]?\s*f\b)", low):
        extracted["sex"] = 0; confidence["sex"] = "high"

    # Blood pressure (systolic from e.g. 148/92 mmHg or Resting BP: 148)
    bp = find_num([
        r"(?:bp|blood\s*pressure|resting\s*bp)\s*[:=\-]?\s*(\d{2,3})\s*/\s*\d{2,3}",
        r"(?:bp|blood\s*pressure|resting\s*bp)\s*[:=\-]?\s*(\d{2,3})\s*(?:mm\s*hg)?",
    ], int, 80, 230)
    if bp is not None:
        extracted["trestbps"] = bp; confidence["trestbps"] = "high"

    chol = find_num([
        r"(?:total\s*cholesterol|serum\s*cholesterol|cholesterol|chol)\s*[:=\-]?\s*(\d{3})\s*(?:mg/dl)?",
    ], int, 100, 500)
    if chol is not None:
        extracted["chol"] = chol; confidence["chol"] = "high"

    fbs_mg = find_num([
        r"(?:fasting\s*blood\s*sugar|fasting\s*glucose|blood\s*sugar|fbs)[^\d\n]{0,16}(\d{2,3})\s*(?:mg/?dl)?",
    ], int, 50, 400)
    if fbs_mg is not None:
        extracted["fbs"] = 1 if fbs_mg > 120 else 0
        confidence["fbs"] = f"high ({fbs_mg} mg/dL)"

    thalach = find_num([
        r"(?:max(?:imum)?\s*heart\s*rate|peak\s*hr|thalach)\s*[:=\-]?\s*(\d{2,3})\s*(?:bpm)?",
    ], int, 60, 220)
    if thalach is not None:
        extracted["thalach"] = thalach; confidence["thalach"] = "high"

    oldpeak = find_num([
        r"(?:st\s*depression(?:\s*\(oldpeak\))?|oldpeak)[^\d\n]{0,14}(\d+(?:\.\d+)?)\s*(?:mm)?",
    ], float, 0.0, 6.5)
    if oldpeak is not None:
        extracted["oldpeak"] = round(oldpeak, 1); confidence["oldpeak"] = "high"

    ca = find_num([r"(?:major\s*vessels|coronary\s*vessels|vessels\s*colored|ca)\s*[:=\-]?\s*([0-3])\b"], int, 0, 3)
    if ca is not None:
        extracted["ca"] = ca; confidence["ca"] = "high"

    # Categorical findings
    if "typical angina" in low:
        extracted["cp"] = 0; confidence["cp"] = "high"
    elif "atypical angina" in low:
        extracted["cp"] = 1; confidence["cp"] = "high"
    elif "non-anginal" in low or "non anginal" in low:
        extracted["cp"] = 2; confidence["cp"] = "medium"
    elif "asymptomatic" in low:
        extracted["cp"] = 3; confidence["cp"] = "medium"

    if re.search(r"exercise[- ]induced\s*angina\s*[:=\-]?\s*(yes|positive|present|1)", low):
        extracted["exang"] = 1; confidence["exang"] = "high"
    elif re.search(r"exercise[- ]induced\s*angina\s*[:=\-]?\s*(no|negative|absent|0)", low):
        extracted["exang"] = 0; confidence["exang"] = "high"

    if "st-t" in low or "st abnormality" in low or "t-wave inversion" in low:
        extracted["restecg"] = 1; confidence["restecg"] = "medium"
    elif "lv hypertrophy" in low or "left ventricular hypertrophy" in low:
        extracted["restecg"] = 2; confidence["restecg"] = "medium"
    elif "normal ecg" in low or "resting ecg: normal" in low:
        extracted["restecg"] = 0; confidence["restecg"] = "medium"

    if "downsloping" in low:
        extracted["slope"] = 0; confidence["slope"] = "medium"
    elif "flat st" in low or "slope: flat" in low:
        extracted["slope"] = 1; confidence["slope"] = "medium"
    elif "upsloping" in low:
        extracted["slope"] = 2; confidence["slope"] = "medium"

    if "reversible defect" in low:
        extracted["thal"] = 3; confidence["thal"] = "high"
    elif "fixed defect" in low:
        extracted["thal"] = 2; confidence["thal"] = "high"
    elif "normal perfusion" in low or "thalassemia: normal" in low:
        extracted["thal"] = 1; confidence["thal"] = "medium"

    # Extract comorbidity & medication cues for longitudinal history
    history_cues = {
        "smoking": bool(re.search(r"\b(smoker|smoking\s*:\s*yes|tobacco\s*use)\b", low)),
        "diabetes": bool(re.search(r"\b(diabetes|diabetic|hba1c\s*[:=]?\s*[6-9]\.\d)\b", low)),
        "hypertension": bool(re.search(r"\b(hypertension|htn)\b", low)),
        "family_history": bool(re.search(r"\b(family\s*history.*(?:cad|cvd|heart|mi))\b", low)),
        "medications": sorted(set(re.findall(
            r"\b(atorvastatin|rosuvastatin|metformin|amlodipine|lisinopril|losartan|aspirin|clopidogrel|metoprolol|bisoprolol)\b",
            low,
        ))),
    }
    return {
        "extracted_features": extracted,
        "confidence": confidence,
        "history_cues": history_cues,
    }


def _call_gemini_vision(image_bytes: bytes, mime_type: str, api_key: str) -> dict | None:
    """Optional online multimodal extraction using Gemini Vision REST API (stdlib urllib)."""
    b64 = base64.b64encode(image_bytes).decode("ascii")
    prompt = (
        "Extract clinical cardiology biomarkers from this medical report image and return ONLY valid JSON "
        "with keys: raw_text (string), age (int), sex (1=male,0=female), cp (0-3), trestbps (int), "
        "chol (int), fbs (0 or 1 if >120 mg/dL), restecg (0-2), thalach (int), exang (0/1), "
        "oldpeak (float), slope (0-2), ca (0-3), thal (1-3), medications (list of strings). "
        "Omit any key not clearly present in the report."
    )
    body = json.dumps({
        "contents": [{
            "parts": [
                {"text": prompt},
                {"inline_data": {"mime_type": mime_type, "data": b64}},
            ]
        }]
    }).encode("utf-8")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={api_key}"
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        text_out = data["candidates"][0]["content"]["parts"][0]["text"]
        m = re.search(r"\{.*\}", text_out, flags=re.DOTALL)
        if m:
            return json.loads(m.group(0))
    except Exception:
        return None
    return None


def analyze_report_payload(
    image_b64: str | None = None,
    raw_text: str | None = None,
    filename: str = "report.png",
    online_mode: bool = False,
) -> dict:
    """
    Analyze an uploaded medical report image (PNG/JPG/WebP), PDF, or pasted clinical text.
    Uses Gemini Vision if online_mode=True and GEMINI_API_KEY is set; otherwise runs
    local Pillow preprocessing + pytesseract/pdfplumber + Clinical Regex Entity Extractor.
    """
    engine_used = "Local Clinical Text Parser"
    quality = {}
    ocr_text = (raw_text or "").strip()

    if image_b64:
        # Strip data URL prefix if present
        if "," in image_b64:
            header, image_b64 = image_b64.split(",", 1)
            mime_match = re.search(r"data:([^;]+);", header)
            mime_type = mime_match.group(1) if mime_match else "image/png"
        else:
            mime_type = "application/pdf" if filename.lower().endswith(".pdf") else "image/png"
        raw_bytes = base64.b64decode(image_b64)

        # 1. Try Online Gemini Vision first if enabled & API key available
        api_key = os.environ.get("GEMINI_API_KEY", "").strip()
        if online_mode and api_key and mime_type.startswith("image/"):
            vision_json = _call_gemini_vision(raw_bytes, mime_type, api_key)
            if vision_json:
                engine_used = "Online Gemini Vision API (gemini-2.5-flash)"
                ocr_text = vision_json.get("raw_text", ocr_text)
                parsed = extract_clinical_entities(ocr_text)
                # Merge structured vision fields with regex fallback
                for k in ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
                          "thalach", "exang", "oldpeak", "slope", "ca", "thal"]:
                    if k in vision_json and vision_json[k] is not None:
                        parsed["extracted_features"][k] = vision_json[k]
                        parsed["confidence"][k] = "vision-api"
                if vision_json.get("medications"):
                    parsed["history_cues"]["medications"] = sorted(set(
                        parsed["history_cues"]["medications"] + [str(m).lower() for m in vision_json["medications"]]
                    ))
                return {
                    "engine": engine_used,
                    "filename": filename,
                    "quality": quality,
                    "ocr_text": ocr_text,
                    **parsed,
                }

        # 2. Local Offline Processing (PDF or Image)
        if mime_type == "application/pdf" or filename.lower().endswith(".pdf"):
            try:
                import pdfplumber
                with pdfplumber.open(io.BytesIO(raw_bytes)) as pdf:
                    ocr_text = "\n".join(p.extract_text() or "" for p in pdf.pages).strip()
                engine_used = "Local Offline PDF Extractor (pdfplumber)"
            except Exception as e:
                engine_used = f"Local PDF Fallback ({type(e).__name__})"
        else:
            proc_img, quality = preprocess_report_image(raw_bytes)
            try:
                import pytesseract
                ocr_text = pytesseract.image_to_string(proc_img).strip()
                engine_used = "Local Offline OCR (Pillow + Tesseract)"
            except Exception:
                # Graceful note when Tesseract binary isn't installed on host
                engine_used = "Local Image Preprocessor (install Tesseract or set GEMINI_API_KEY for full OCR)"
                if not ocr_text:
                    # Check if PNG carries embedded clinical text metadata (e.g., our bundled sample report)
                    orig = Image.open(io.BytesIO(raw_bytes))
                    embedded = orig.info.get("clinical_report_text", "")
                    if embedded:
                        ocr_text = embedded
                        engine_used = "Local Offline Report Reader (Embedded Lab Text + Regex Parser)"

    parsed = extract_clinical_entities(ocr_text)
    return {
        "engine": engine_used,
        "filename": filename,
        "quality": quality,
        "ocr_text": ocr_text,
        **parsed,
    }


def generate_sample_lab_report_png(out_path: str | Path) -> Path:
    """Create a clean, realistic synthetic cardiology lab report PNG for live demos."""
    from PIL import PngImagePlugin
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    report_lines = [
        "CARDIOLOGY DIAGNOSTIC & STRESS ECG REPORT",
        "Patient MRN: CP-1042   |   Age: 62 yrs   |   Sex: Male",
        "---------------------------------------------------------------",
        "CLINICAL VITALS & LIPID PANEL:",
        "  Resting Blood Pressure : 152/94 mmHg  (Hypertension history)",
        "  Total Cholesterol      : 286 mg/dL    (Elevated)",
        "  Fasting Blood Sugar    : 138 mg/dL    (Diabetes mellitus)",
        "  Chest Pain Presentation: Typical Angina",
        "---------------------------------------------------------------",
        "EXERCISE STRESS TEST & IMAGING:",
        "  Resting ECG            : ST-T abnormality",
        "  Max Heart Rate         : 124 bpm",
        "  Exercise-Induced Angina: Yes",
        "  ST Depression (oldpeak): 2.3 mm  |  Slope: Flat ST",
        "  Major Vessels Colored  : 2",
        "  Thalassemia Perfusion  : Reversible defect",
        "---------------------------------------------------------------",
        "ACTIVE MEDICATIONS: Atorvastatin 20mg, Metformin 500mg, Amlodipine 5mg",
        "HISTORY NOTES: Smoking: Yes | Family history of premature CAD",
    ]
    full_text = "\n".join(report_lines)

    img = Image.new("RGB", (900, 620), color=(252, 253, 255))
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, 900, 68], fill=(15, 76, 92))
    try:
        font_hdr = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 22)
        font_body = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 17)
    except Exception:
        font_hdr = ImageFont.load_default()
        font_body = ImageFont.load_default()

    draw.text((28, 22), "METRO HEART LAB — DIAGNOSTIC REPORT (DEMO)", fill=(255, 255, 255), font=font_hdr)
    y = 92
    for line in report_lines:
        color = (15, 76, 92) if line.startswith(("CLINICAL", "EXERCISE", "ACTIVE")) else (30, 41, 59)
        draw.text((32, y), line, fill=color, font=font_body)
        y += 27

    meta = PngImagePlugin.PngInfo()
    meta.add_text("clinical_report_text", full_text)
    img.save(out_path, "PNG", pnginfo=meta)
    return out_path
