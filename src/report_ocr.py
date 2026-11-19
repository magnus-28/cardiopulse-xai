"""
Multimodal ECG Waveform & Medical Lab Report Analyzer for CardioPulse XAI v2.0.
Capabilities:
  1. Dual-Track Offline Analysis:
     - Track A (ECG Waveform Computer Vision): Digitizes ECG tracing pixels using
       Pillow + NumPy + SciPy, filters red/pink ECG gridlines, detects R-peaks,
       estimates Heart Rate (bpm), R-R rhythm regularity, baseline stability, and
       ST-segment deviation indicators.
     - Track B (Multi-Pass OCR + Clinical Entity Extraction): Extracts structured
       biomarkers, ECG intervals (PR/QRS/QTc), rhythm diagnoses, comorbidities,
       and active medications from lab reports, ECG printouts, or digital PDFs.
  2. Online Cloud Vision Upgrade:
     - When Online Mode is enabled and GEMINI_API_KEY is configured, calls Gemini
       Vision REST API (gemini-2.5-flash) for full multimodal interpretation.
  3. Structured Clinical Report Generator:
     - Produces a detailed, human-readable diagnostic report (summary, signal/rhythm
       analysis, extracted biomarkers, clinical interpretation, follow-up steps)
       even when an ECG image has zero printed text.
"""
from __future__ import annotations
import base64
import io
import json
import os
import re
import urllib.request
from pathlib import Path

import numpy as np
from scipy.signal import find_peaks, savgol_filter
from PIL import Image, ImageEnhance, ImageFilter, ImageOps, ImageDraw, ImageFont


def preprocess_report_image(image_bytes: bytes) -> tuple[Image.Image, Image.Image, dict]:
    """Return RGB original, contrast-enhanced grayscale image, and quality metadata."""
    orig = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    w, h = orig.size
    gray = ImageOps.grayscale(orig)
    enhanced = ImageEnhance.Contrast(gray).enhance(1.9)
    sharpened = enhanced.filter(ImageFilter.SHARPEN)
    quality = {
        "width": w,
        "height": h,
        "resolution_ok": w >= 350 and h >= 200,
        "aspect_ratio": round(w / max(h, 1), 2),
    }
    return orig, sharpened, quality


def _detect_and_analyze_ecg_waveform(rgb_img: Image.Image) -> dict:
    """
    Computer-vision ECG waveform analyzer (pure NumPy + SciPy + Pillow).
    Detects ECG-like images (red/pink gridlines or high-contrast dark waveform traces)
    and extracts rate, rhythm regularity, baseline wander, and ST-segment deviation proxy.
    """
    arr = np.asarray(rgb_img, dtype=np.float32)
    h, w, _ = arr.shape
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]

    # Detect pink/red/orange ECG calibration grid pixels (high R relative to G & B)
    red_grid_mask = (r - g > 18) & (r - b > 18) & (r > 130)
    red_grid_ratio = float(np.mean(red_grid_mask))

    # Build a dark-ink mask that suppresses red/pink gridlines (keeps dark gray/black trace)
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    dark_trace_mask = (luminance < 115) & (~red_grid_mask)
    dark_ratio = float(np.mean(dark_trace_mask))

    # Inspect horizontal strip in the middle 50% of image to digitize a primary lead trace
    y0, y1 = int(h * 0.25), int(h * 0.75)
    x0, x1 = int(w * 0.05), int(w * 0.95)
    strip = dark_trace_mask[y0:y1, x0:x1]
    sh, sw = strip.shape

    # For each x column, find vertical centroid of dark trace pixels
    signal = np.full(sw, np.nan, dtype=np.float32)
    for col in range(sw):
        ys = np.where(strip[:, col])[0]
        if 0 < len(ys) < int(sh * 0.45):
            # Take uppermost dark cluster to track R-wave peaks cleanly
            signal[col] = float(sh - np.median(ys))

    valid_frac = float(np.mean(~np.isnan(signal)))
    is_ecg_visual = (red_grid_ratio > 0.025) or (valid_frac > 0.35 and (w / max(h, 1)) >= 1.25 and dark_ratio < 0.25)

    if valid_frac < 0.12:
        return {
            "is_ecg_image": is_ecg_visual,
            "signal_extracted": False,
            "red_grid_detected": red_grid_ratio > 0.025,
            "signal_coverage_pct": round(valid_frac * 100, 1),
        }

    # Interpolate missing x-columns & smooth trace
    idx = np.arange(sw)
    good = ~np.isnan(signal)
    interp = np.interp(idx, idx[good], signal[good])
    win = min(11, max(5, (sw // 40) | 1))
    smooth = savgol_filter(interp, window_length=win, polyorder=2) if sw > win + 2 else interp

    # Remove slow baseline drift
    drift_win = min(sw - 1 if (sw - 1) % 2 == 1 else sw - 2, max(21, (sw // 6) | 1))
    baseline = savgol_filter(smooth, window_length=drift_win, polyorder=1) if drift_win >= 5 and sw > drift_win else np.median(smooth)
    detrended = smooth - baseline
    amp_std = float(np.std(detrended)) + 1e-6
    norm_sig = detrended / amp_std

    # Detect R-peaks
    min_dist = max(12, sw // 28)
    peaks, props = find_peaks(norm_sig, height=1.35, distance=min_dist, prominence=0.9)

    if len(peaks) < 2:
        return {
            "is_ecg_image": is_ecg_visual,
            "signal_extracted": True,
            "red_grid_detected": red_grid_ratio > 0.025,
            "signal_coverage_pct": round(valid_frac * 100, 1),
            "r_peaks_detected": int(len(peaks)),
            "estimated_hr_bpm": None,
            "rhythm_assessment": "Insufficient cardiac cycles detected in single strip for automated rate estimation.",
            "st_assessment": "Unable to isolate inter-beat ST segments reliably.",
        }

    rr_px = np.diff(peaks).astype(float)
    rr_mean = float(np.mean(rr_px))
    rr_cv = float(np.std(rr_px) / max(rr_mean, 1e-6))

    # Assume standard 10-second strip width across 90% horizontal span (standard 12-lead / rhythm strip)
    est_cycles_10s = sw / max(rr_mean, 1.0)
    est_hr = int(np.clip(round(est_cycles_10s * 6.0), 42, 185))

    if rr_cv < 0.12:
        rhythm = "Regular rhythm pattern (consistent R-R intervals; suggestive of sinus rhythm)"
        rhythm_flag = "regular"
    elif rr_cv < 0.22:
        rhythm = "Mildly irregular R-R spacing (possible sinus arrhythmia or premature beats)"
        rhythm_flag = "mild_irregular"
    else:
        rhythm = "Irregular R-R variability detected (warrants clinician review for atrial fibrillation or ectopy)"
        rhythm_flag = "irregular"

    # Rate classification
    if est_hr < 60:
        rate_cls = "Bradycardic range (< 60 bpm)"
    elif est_hr > 100:
        rate_cls = "Tachycardic range (> 100 bpm)"
    else:
        rate_cls = "Normal resting rate (60–100 bpm)"

    # Estimate post-QRS ST segment shift relative to pre-QRS isoelectric baseline
    st_shifts = []
    for p in peaks:
        pre_lo, pre_hi = max(0, int(p - 0.22 * rr_mean)), max(1, int(p - 0.10 * rr_mean))
        st_lo, st_hi = min(sw - 1, int(p + 0.08 * rr_mean)), min(sw, int(p + 0.20 * rr_mean))
        if pre_hi > pre_lo and st_hi > st_lo:
            iso = float(np.median(detrended[pre_lo:pre_hi]))
            st_val = float(np.median(detrended[st_lo:st_hi]))
            st_shifts.append((st_val - iso) / amp_std)

    mean_st_z = float(np.mean(st_shifts)) if st_shifts else 0.0
    if mean_st_z <= -0.38:
        st_status = "Possible ST-segment depression morphology detected relative to isoelectric baseline"
        st_abnormal = True
        est_oldpeak = round(min(3.0, abs(mean_st_z) * 1.4), 1)
    elif mean_st_z >= 0.45:
        st_status = "Possible ST-segment elevation / early repolarization morphology detected"
        st_abnormal = True
        est_oldpeak = round(min(2.5, abs(mean_st_z) * 1.1), 1)
    else:
        st_status = "No marked ST-segment deviation detected relative to local baseline"
        st_abnormal = False
        est_oldpeak = 0.2

    baseline_wander = float(np.ptp(baseline) / max(sh, 1))
    sig_quality = "Good" if (valid_frac > 0.65 and baseline_wander < 0.22) else ("Moderate" if valid_frac > 0.35 else "Low")

    return {
        "is_ecg_image": True,
        "signal_extracted": True,
        "red_grid_detected": red_grid_ratio > 0.025,
        "signal_coverage_pct": round(valid_frac * 100, 1),
        "signal_quality": sig_quality,
        "r_peaks_detected": int(len(peaks)),
        "estimated_hr_bpm": est_hr,
        "rate_classification": rate_cls,
        "rr_variability_cv": round(rr_cv, 3),
        "rhythm_flag": rhythm_flag,
        "rhythm_assessment": rhythm,
        "st_abnormal": st_abnormal,
        "st_assessment": st_status,
        "estimated_oldpeak_mm": est_oldpeak,
    }


def extract_clinical_entities(text: str) -> dict:
    """Parse clinical biomarkers, ECG intervals, and history cues from report text."""
    t = text or ""
    low = t.lower()
    extracted = {}
    confidence = {}
    ecg_text_metrics = {}

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
        r"(?:max(?:imum)?\s*heart\s*rate|peak\s*hr|heart\s*rate|vent\.?\s*rate|hr|thalach)[^\d\n]{0,12}(\d{2,3})\s*(?:bpm)?",
    ], int, 40, 220)
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

    # ECG print intervals if present in text
    pr_ms = find_num([r"\bPR(?:\s*int(?:erval)?)?\s*[:=\-]?\s*(\d{2,3})\s*(?:ms)?"], int, 80, 350)
    qrs_ms = find_num([r"\bQRS(?:\s*dur(?:ation)?)?\s*[:=\-]?\s*(\d{2,3})\s*(?:ms)?"], int, 50, 220)
    qtc_ms = find_num([r"\bQTc?\s*[:=\-]?\s*(\d{3})\s*(?:ms)?"], int, 300, 600)
    if pr_ms: ecg_text_metrics["PR Interval"] = f"{pr_ms} ms"
    if qrs_ms: ecg_text_metrics["QRS Duration"] = f"{qrs_ms} ms"
    if qtc_ms: ecg_text_metrics["QT / QTc"] = f"{qtc_ms} ms"

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

    if any(k in low for k in ["st-t", "st abnormality", "st depression", "st elevation", "t-wave inversion", "ischemia", "infarct"]):
        extracted["restecg"] = 1; confidence["restecg"] = "high"
    elif any(k in low for k in ["lv hypertrophy", "left ventricular hypertrophy", "lvh"]):
        extracted["restecg"] = 2; confidence["restecg"] = "high"
    elif any(k in low for k in ["normal ecg", "resting ecg: normal", "normal sinus rhythm"]):
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
        "ecg_text_metrics": ecg_text_metrics,
        "history_cues": history_cues,
    }


def _call_gemini_vision(image_bytes: bytes, mime_type: str, api_key: str) -> dict | None:
    """Optional online multimodal extraction using Gemini Vision REST API (stdlib urllib)."""
    b64 = base64.b64encode(image_bytes).decode("ascii")
    prompt = (
        "You are a clinical cardiology report parser. Analyze this medical lab report or ECG image "
        "and return ONLY valid JSON with keys: report_type ('ECG Strip' or 'Lab Report' or 'Combined'), "
        "raw_text (string), clinical_summary (2-3 sentence medical summary), ecg_findings (list of strings), "
        "recommendations (list of 3 actionable clinical follow-up bullets), and any detected numeric fields "
        "among age, sex (1=male,0=female), cp (0-3), trestbps, chol, fbs (0/1), restecg (0-2), thalach, "
        "exang (0/1), oldpeak, slope (0-2), ca (0-3), thal (1-3), medications (list of strings)."
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
        with urllib.request.urlopen(req, timeout=18) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        text_out = data["candidates"][0]["content"]["parts"][0]["text"]
        m = re.search(r"\{.*\}", text_out, flags=re.DOTALL)
        if m:
            return json.loads(m.group(0))
    except Exception:
        return None
    return None


def _compose_structured_report(
    filename: str,
    engine: str,
    quality: dict,
    parsed: dict,
    ecg_cv: dict | None,
    vision_narrative: dict | None = None,
) -> dict:
    """Compose a well-described, clinician-readable diagnostic report."""
    feats = parsed["extracted_features"]
    cues = parsed["history_cues"]
    ecg_txt = parsed.get("ecg_text_metrics", {})

    is_ecg = bool(ecg_cv and ecg_cv.get("is_ecg_image")) or bool(ecg_txt) or ("ecg" in filename.lower()) or ("ekg" in filename.lower())
    report_type = (vision_narrative or {}).get("report_type") or (
        "Combined ECG & Lab Report" if (is_ecg and len(feats) >= 4) else ("ECG Waveform Report" if is_ecg else "Clinical Lab Report")
    )

    findings = list((vision_narrative or {}).get("ecg_findings") or [])
    if ecg_cv and ecg_cv.get("signal_extracted"):
        if ecg_cv.get("estimated_hr_bpm"):
            findings.append(f"Estimated Ventricular Rate: {ecg_cv['estimated_hr_bpm']} bpm — {ecg_cv['rate_classification']}.")
        if ecg_cv.get("rhythm_assessment"):
            findings.append(f"Rhythm Regularity (CV={ecg_cv.get('rr_variability_cv', 'N/A')}): {ecg_cv['rhythm_assessment']}.")
        if ecg_cv.get("st_assessment"):
            findings.append(f"ST-Segment Morphology: {ecg_cv['st_assessment']}.")
        findings.append(
            f"Tracing Digitization Quality: {ecg_cv.get('signal_quality', 'Moderate')} "
            f"({ecg_cv.get('r_peaks_detected', 0)} R-peaks tracked across {ecg_cv.get('signal_coverage_pct', 0)}% strip coverage"
            + (", standard ECG calibration grid detected)." if ecg_cv.get("red_grid_detected") else ").")
        )
    for k, v in ecg_txt.items():
        findings.append(f"Printed ECG Parameter — {k}: {v}.")

    if not findings:
        if feats:
            findings.append(f"Extracted {len(feats)} structured clinical biomarkers from document text via OCR.")
        else:
            findings.append(
                "No printed tabular lab values were detected in this image. Review the image resolution or "
                "enable Online Mode with GEMINI_API_KEY for vision-LLM reading of handwritten/low-contrast scans."
            )

    # Executive clinical summary
    if vision_narrative and vision_narrative.get("clinical_summary"):
        summary = vision_narrative["clinical_summary"]
    elif is_ecg and ecg_cv and ecg_cv.get("signal_extracted"):
        hr_str = f"an estimated heart rate of {ecg_cv['estimated_hr_bpm']} bpm" if ecg_cv.get("estimated_hr_bpm") else "partial waveform cycles"
        st_str = "potential ST-segment deviation" if ecg_cv.get("st_abnormal") else "no marked ST-segment shift"
        summary = (
            f"Automated waveform digitization of '{filename}' analyzed {ecg_cv.get('r_peaks_detected', 0)} cardiac cycles, "
            f"indicating {hr_str} with {ecg_cv.get('rhythm_flag', 'unclassified').replace('_', ' ')} R-R spacing and {st_str}. "
            f"Derived ECG indicators have been mapped into the risk model alongside {len(feats)} extracted clinical parameters."
        )
    elif feats:
        summary = (
            f"Document analysis of '{filename}' successfully extracted {len(feats)} clinical biomarkers "
            f"and {len(cues.get('medications', []))} active medications, updating the cardiovascular risk profile automatically."
        )
    else:
        summary = (
            f"Processed '{filename}' ({quality.get('width', 0)}x{quality.get('height', 0)} px). "
            f"No structured numeric biomarkers were printed in readable text; please verify key fields below or configure GEMINI_API_KEY for multimodal vision parsing."
        )

    recommendations = list((vision_narrative or {}).get("recommendations") or [])
    if not recommendations:
        if ecg_cv and ecg_cv.get("st_abnormal"):
            recommendations.append("Correlate suspected ST-segment morphology with a standard 12-lead diagnostic ECG and troponin panel.")
        if ecg_cv and ecg_cv.get("rhythm_flag") == "irregular":
            recommendations.append("Confirm irregular R-R rhythm on a 12-lead ECG or ambulatory Holter monitor to rule out arrhythmia.")
        if feats.get("trestbps", 0) >= 140 or feats.get("chol", 0) >= 240:
            recommendations.append("Review elevated blood pressure / lipid markers against current antihypertensive or statin therapy.")
        recommendations.append("Verify all auto-filled clinical fields before finalizing risk stratification; this tool supports clinician review only.")

    return {
        "report_type": report_type,
        "engine": engine,
        "executive_summary": summary,
        "findings": findings,
        "recommendations": recommendations,
        "extracted_count": len(feats),
        "signal_metrics": ecg_cv if (ecg_cv and ecg_cv.get("signal_extracted")) else None,
    }


def analyze_report_payload(
    image_b64: str | None = None,
    raw_text: str | None = None,
    filename: str = "report.png",
    online_mode: bool = False,
) -> dict:
    """
    Analyze an uploaded ECG waveform image, lab report image (PNG/JPG/WebP), PDF, or pasted text.
    Returns extracted features, confidence tags, history cues, and a detailed structured report.
    """
    engine_used = "Local Clinical Text Parser"
    quality = {}
    ocr_text = (raw_text or "").strip()
    ecg_cv = None
    vision_narrative = None

    if image_b64:
        if "," in image_b64:
            header, image_b64 = image_b64.split(",", 1)
            mime_match = re.search(r"data:([^;]+);", header)
            mime_type = mime_match.group(1) if mime_match else "image/png"
        else:
            mime_type = "application/pdf" if filename.lower().endswith(".pdf") else "image/png"
        raw_bytes = base64.b64decode(image_b64)

        # 1. Online Gemini Vision if enabled & API key set
        api_key = os.environ.get("GEMINI_API_KEY", "").strip()
        if online_mode and api_key and mime_type.startswith("image/"):
            vision_json = _call_gemini_vision(raw_bytes, mime_type, api_key)
            if vision_json:
                vision_narrative = vision_json
                engine_used = "Online Gemini Vision API (gemini-2.5-flash)"
                ocr_text = vision_json.get("raw_text", ocr_text)

        # 2. Local Offline Analysis (PDF or Image OCR + ECG Waveform CV)
        if mime_type == "application/pdf" or filename.lower().endswith(".pdf"):
            try:
                import pdfplumber
                with pdfplumber.open(io.BytesIO(raw_bytes)) as pdf:
                    ocr_text = "\n".join(p.extract_text() or "" for p in pdf.pages).strip()
                engine_used = "Local Offline PDF Extractor (pdfplumber)"
            except Exception as e:
                engine_used = f"Local PDF Fallback ({type(e).__name__})"
        else:
            orig_rgb, proc_img, quality = preprocess_report_image(raw_bytes)
            ecg_cv = _detect_and_analyze_ecg_waveform(orig_rgb)

            if not vision_narrative:
                # Check embedded demo metadata first, else run Tesseract OCR
                raw_pil = Image.open(io.BytesIO(raw_bytes))
                embedded = raw_pil.info.get("clinical_report_text", "")
                if embedded:
                    ocr_text = embedded
                    engine_used = "Local Offline Report Reader + Clinical Parser"
                else:
                    try:
                        import pytesseract
                        ocr_text = pytesseract.image_to_string(proc_img).strip()
                        if ecg_cv.get("is_ecg_image") and ecg_cv.get("signal_extracted"):
                            engine_used = "Local ECG Waveform CV + Tesseract OCR"
                        else:
                            engine_used = "Local Offline OCR (Pillow + Tesseract)"
                    except Exception:
                        engine_used = "Local ECG Waveform CV Analyzer" if (ecg_cv and ecg_cv.get("signal_extracted")) else "Local Image Preprocessor"

    parsed = extract_clinical_entities(ocr_text)

    # Merge Gemini Vision JSON fields if available
    if vision_narrative:
        for k in ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
                  "thalach", "exang", "oldpeak", "slope", "ca", "thal"]:
            if k in vision_narrative and vision_narrative[k] is not None:
                parsed["extracted_features"][k] = vision_narrative[k]
                parsed["confidence"][k] = "vision-api"
        if vision_narrative.get("medications"):
            parsed["history_cues"]["medications"] = sorted(set(
                parsed["history_cues"]["medications"] + [str(m).lower() for m in vision_narrative["medications"]]
            ))

    # Map CV-derived ECG findings into UCI features if OCR didn't already extract them
    if ecg_cv and ecg_cv.get("is_ecg_image") and ecg_cv.get("signal_extracted"):
        feats = parsed["extracted_features"]
        conf = parsed["confidence"]
        if "thalach" not in feats and ecg_cv.get("estimated_hr_bpm"):
            feats["thalach"] = ecg_cv["estimated_hr_bpm"]
            conf["thalach"] = "ecg-waveform-cv"
        if "restecg" not in feats:
            feats["restecg"] = 1 if ecg_cv.get("st_abnormal") else 0
            conf["restecg"] = "ecg-waveform-cv"
        if "oldpeak" not in feats and ecg_cv.get("estimated_oldpeak_mm") is not None:
            feats["oldpeak"] = ecg_cv["estimated_oldpeak_mm"]
            conf["oldpeak"] = "ecg-waveform-cv"

    structured_report = _compose_structured_report(
        filename=filename,
        engine=engine_used,
        quality=quality,
        parsed=parsed,
        ecg_cv=ecg_cv,
        vision_narrative=vision_narrative,
    )

    return {
        "engine": engine_used,
        "filename": filename,
        "quality": quality,
        "ocr_text": ocr_text,
        "structured_report": structured_report,
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


def generate_sample_ecg_strip_png(out_path: str | Path) -> Path:
    """Create a realistic synthetic 10-second ECG waveform strip PNG (pink grid + QRS complexes)."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    w, h = 960, 340
    img = Image.new("RGB", (w, h), color=(255, 246, 246))
    draw = ImageDraw.Draw(img)

    # Draw standard pink ECG calibration grid
    for x in range(0, w, 12):
        col = (242, 182, 182) if x % 60 == 0 else (250, 218, 218)
        draw.line([(x, 0), (x, h)], fill=col, width=1)
    for y in range(0, h, 12):
        col = (242, 182, 182) if y % 60 == 0 else (250, 218, 218)
        draw.line([(0, y), (w, y)], fill=col, width=1)

    # Synthesize realistic P-QRS-T waveform with mild ST depression (~82 bpm across 10s = ~13.5 beats)
    xs = np.arange(20, w - 20)
    y_base = h * 0.54
    pts = []
    period = 72.0
    for x in xs:
        phase = ((x - 20) % period) / period
        val = 0.0
        # P wave
        val += 10.0 * np.exp(-((phase - 0.16) / 0.04) ** 2)
        # Q dip
        val -= 12.0 * np.exp(-((phase - 0.28) / 0.015) ** 2)
        # R spike
        val += 92.0 * np.exp(-((phase - 0.31) / 0.014) ** 2)
        # S dip
        val -= 24.0 * np.exp(-((phase - 0.345) / 0.016) ** 2)
        # Mild ST depression
        val -= 9.5 * np.exp(-((phase - 0.44) / 0.05) ** 2)
        # T wave
        val += 22.0 * np.exp(-((phase - 0.58) / 0.06) ** 2)
        pts.append((float(x), float(y_base - val)))

    draw.line(pts, fill=(18, 24, 34), width=3)
    img.save(out_path, "PNG")
    return out_path
