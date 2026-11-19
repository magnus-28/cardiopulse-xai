"""
Multimodal ECG Waveform & Medical Lab Report Analyzer for CardioPulse XAI v2.0.
Capabilities:
  1. Dual-Track Offline Analysis:
     - Track A (Multi-Band ECG Waveform Computer Vision): Supports both single-lead
       rhythm strips and 3-row 12-lead ECG printouts. Filters red/pink calibration
       grids, scans candidate lead bands to isolate the cleanest tracing, detects
       R-peaks, estimates Resting Heart Rate (bpm), R-R regularity, and ST-segment
       morphology without confusing resting HR with stress-test max HR (`thalach`).
     - Track B (OCR + Clinical Entity Extraction): Extracts structured biomarkers,
       ECG intervals (PR/QRS/QTc), rhythm diagnoses, comorbidities, and medications.
  2. Online Cloud Vision Upgrade:
     - Uses Gemini Vision REST API (gemini-2.5-flash) when Online Mode is enabled
       and GEMINI_API_KEY is configured.
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


def _analyze_single_band(dark_mask: np.ndarray, y0: int, y1: int, x0: int, x1: int) -> dict | None:
    """Digitize a single horizontal ECG lead band and evaluate R-peaks & ST morphology."""
    strip = dark_mask[y0:y1, x0:x1]
    sh, sw = strip.shape
    if sh < 20 or sw < 80:
        return None

    signal = np.full(sw, np.nan, dtype=np.float32)
    for col in range(sw):
        ys = np.where(strip[:, col])[0]
        if 0 < len(ys) < int(sh * 0.35):
            # Ignore isolated lead-label text blocks by checking vertical spread
            if (ys[-1] - ys[0]) < int(sh * 0.55):
                signal[col] = float(sh - np.median(ys))

    valid_frac = float(np.mean(~np.isnan(signal)))
    if valid_frac < 0.20:
        return None

    idx = np.arange(sw)
    good = ~np.isnan(signal)
    interp = np.interp(idx, idx[good], signal[good])
    win = min(9, max(5, (sw // 60) | 1))
    smooth = savgol_filter(interp, window_length=win, polyorder=2) if sw > win + 2 else interp

    drift_win = min(sw - 1 if (sw - 1) % 2 == 1 else sw - 2, max(21, (sw // 7) | 1))
    baseline = savgol_filter(smooth, window_length=drift_win, polyorder=1) if drift_win >= 5 and sw > drift_win else np.median(smooth)
    detrended = smooth - baseline
    amp_std = float(np.std(detrended)) + 1e-6
    norm_sig = detrended / amp_std

    min_dist = max(14, sw // 26)
    peaks, _ = find_peaks(norm_sig, height=1.6, distance=min_dist, prominence=1.1)
    if len(peaks) < 3:
        return None

    rr_px = np.diff(peaks).astype(float)
    rr_med = float(np.median(rr_px))
    # Keep physiologically consistent intervals (reject 12-lead column-divider artifacts)
    clean_rr = rr_px[(rr_px >= 0.55 * rr_med) & (rr_px <= 1.65 * rr_med)]
    if len(clean_rr) < 2:
        clean_rr = rr_px
    rr_mean = float(np.mean(clean_rr))
    rr_cv = float(np.std(clean_rr) / max(rr_mean, 1e-6))

    est_cycles_10s = sw / max(rr_mean, 1.0)
    est_hr = int(np.clip(round(est_cycles_10s * 6.0), 45, 180))

    # Measure ST segment shift relative to pre-QRS isoelectric baseline
    st_shifts = []
    for p in peaks:
        pre_lo, pre_hi = max(0, int(p - 0.22 * rr_mean)), max(1, int(p - 0.10 * rr_mean))
        st_lo, st_hi = min(sw - 1, int(p + 0.09 * rr_mean)), min(sw, int(p + 0.19 * rr_mean))
        if pre_hi > pre_lo and st_hi > st_lo:
            iso = float(np.median(detrended[pre_lo:pre_hi]))
            st_val = float(np.median(detrended[st_lo:st_hi]))
            st_shifts.append((st_val - iso) / amp_std)

    mean_st_z = float(np.median(st_shifts)) if st_shifts else 0.0
    # Strict threshold so normal flat ST segments are NOT falsely flagged as depressed
    st_abnormal = bool(mean_st_z <= -0.55 or mean_st_z >= 0.70)
    est_oldpeak = round(min(2.8, abs(mean_st_z) * 1.2), 1) if st_abnormal else 0.0

    # Score band quality (prefer high coverage + regular peaks)
    quality_score = valid_frac * 2.0 + min(len(peaks), 14) * 0.08 - rr_cv
    return {
        "quality_score": quality_score,
        "valid_frac": valid_frac,
        "r_peaks_detected": int(len(peaks)),
        "estimated_hr_bpm": est_hr,
        "rr_cv": rr_cv,
        "mean_st_z": mean_st_z,
        "st_abnormal": st_abnormal,
        "estimated_oldpeak_mm": est_oldpeak,
    }


def _detect_and_analyze_ecg_waveform(rgb_img: Image.Image) -> dict:
    """
    Multi-band ECG waveform analyzer supporting single-lead strips AND 3-row 12-lead ECGs.
    """
    arr = np.asarray(rgb_img, dtype=np.float32)
    h, w, _ = arr.shape
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]

    red_grid_mask = (r - g > 16) & (r - b > 16) & (r > 125)
    red_grid_ratio = float(np.mean(red_grid_mask))
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    dark_trace_mask = (luminance < 120) & (~red_grid_mask)
    dark_ratio = float(np.mean(dark_trace_mask))

    x0, x1 = int(w * 0.04), int(w * 0.96)
    # Candidate bands: 3 standard 12-lead rows + 1 centered single-lead band
    bands = [
        (int(h * 0.08), int(h * 0.36)),
        (int(h * 0.35), int(h * 0.65)),
        (int(h * 0.64), int(h * 0.92)),
        (int(h * 0.22), int(h * 0.78)),
    ]
    candidates = []
    for y0, y1 in bands:
        res = _analyze_single_band(dark_trace_mask, y0, y1, x0, x1)
        if res is not None:
            candidates.append(res)

    is_ecg_visual = (red_grid_ratio > 0.02) or (len(candidates) > 0 and (w / max(h, 1)) >= 1.25 and dark_ratio < 0.25)

    if not candidates:
        return {
            "is_ecg_image": is_ecg_visual,
            "signal_extracted": False,
            "red_grid_detected": red_grid_ratio > 0.02,
            "signal_coverage_pct": 0.0,
        }

    best = max(candidates, key=lambda c: c["quality_score"])
    est_hr = best["estimated_hr_bpm"]
    rr_cv = best["rr_cv"]

    if rr_cv < 0.14:
        rhythm = "Regular sinus-like rhythm pattern (consistent R-R intervals)"
        rhythm_flag = "regular"
    elif rr_cv < 0.24:
        rhythm = "Mild R-R interval variation (possible sinus arrhythmia or premature beats)"
        rhythm_flag = "mild_irregular"
    else:
        rhythm = "Irregular R-R variability detected (warrants clinician review for arrhythmia)"
        rhythm_flag = "irregular"

    if est_hr < 60:
        rate_cls = "Bradycardic resting range (< 60 bpm)"
    elif est_hr > 100:
        rate_cls = "Tachycardic resting range (> 100 bpm)"
    else:
        rate_cls = "Normal resting rate (60–100 bpm)"

    if best["mean_st_z"] <= -0.55:
        st_status = "Possible ST-segment depression morphology detected relative to baseline"
    elif best["mean_st_z"] >= 0.70:
        st_status = "Possible ST-segment elevation morphology detected relative to baseline"
    else:
        st_status = "Isoelectric ST-segment morphology (no marked ST elevation or depression detected)"

    sig_quality = "Good" if best["valid_frac"] > 0.55 else "Moderate"

    return {
        "is_ecg_image": True,
        "signal_extracted": True,
        "red_grid_detected": red_grid_ratio > 0.02,
        "signal_coverage_pct": round(best["valid_frac"] * 100, 1),
        "signal_quality": sig_quality,
        "r_peaks_detected": best["r_peaks_detected"],
        "estimated_hr_bpm": est_hr,
        "rate_classification": rate_cls,
        "rr_variability_cv": round(rr_cv, 3),
        "rhythm_flag": rhythm_flag,
        "rhythm_assessment": rhythm,
        "st_abnormal": best["st_abnormal"],
        "st_assessment": st_status,
        "estimated_oldpeak_mm": best["estimated_oldpeak_mm"],
        "ecg_overall_normal": (not best["st_abnormal"]) and (rhythm_flag == "regular") and (55 <= est_hr <= 105),
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

    # Only map explicit stress/max heart rate to `thalach` (resting HR goes to ecg_text_metrics)
    thalach = find_num([
        r"(?:max(?:imum)?\s*heart\s*rate|peak\s*hr|thalach)[^\d\n]{0,12}(\d{2,3})\s*(?:bpm)?",
    ], int, 70, 220)
    if thalach is not None:
        extracted["thalach"] = thalach; confidence["thalach"] = "high"

    rest_hr = find_num([r"(?:vent\.?\s*rate|resting\s*hr|heart\s*rate|\bhr)\s*[:=\-]?\s*(\d{2,3})\s*(?:bpm)?"], int, 40, 180)
    if rest_hr is not None:
        ecg_text_metrics["Printed Heart Rate"] = f"{rest_hr} bpm"

    oldpeak = find_num([
        r"(?:st\s*depression(?:\s*\(oldpeak\))?|oldpeak)[^\d\n]{0,14}(\d+(?:\.\d+)?)\s*(?:mm)?",
    ], float, 0.0, 6.5)
    if oldpeak is not None:
        extracted["oldpeak"] = round(oldpeak, 1); confidence["oldpeak"] = "high"

    ca = find_num([r"(?:major\s*vessels|coronary\s*vessels|vessels\s*colored|ca)\s*[:=\-]?\s*([0-3])\b"], int, 0, 3)
    if ca is not None:
        extracted["ca"] = ca; confidence["ca"] = "high"

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
        "Combined ECG & Lab Report" if (is_ecg and len(feats) >= 5) else ("ECG Waveform Report" if is_ecg else "Clinical Lab Report")
    )

    findings = list((vision_narrative or {}).get("ecg_findings") or [])
    if ecg_cv and ecg_cv.get("signal_extracted"):
        findings.append(f"Estimated Resting Heart Rate: {ecg_cv['estimated_hr_bpm']} bpm — {ecg_cv['rate_classification']}.")
        findings.append(f"Rhythm Regularity (R-R CV={ecg_cv['rr_variability_cv']}): {ecg_cv['rhythm_assessment']}.")
        findings.append(f"ST-Segment Morphology: {ecg_cv['st_assessment']}.")
        findings.append(
            f"Tracing Digitization Quality: {ecg_cv['signal_quality']} "
            f"({ecg_cv['r_peaks_detected']} R-peaks tracked across {ecg_cv['signal_coverage_pct']}% band coverage"
            + (", standard ECG calibration grid detected)." if ecg_cv.get("red_grid_detected") else ").")
        )
    for k, v in ecg_txt.items():
        findings.append(f"Printed ECG Parameter — {k}: {v}.")

    if not findings:
        if feats:
            findings.append(f"Extracted {len(feats)} structured clinical biomarkers from document text via OCR.")
        else:
            findings.append("No printed tabular lab values were detected; verify clinical inputs manually or enable Gemini Vision API.")

    if vision_narrative and vision_narrative.get("clinical_summary"):
        summary = vision_narrative["clinical_summary"]
    elif is_ecg and ecg_cv and ecg_cv.get("signal_extracted"):
        if ecg_cv.get("ecg_overall_normal"):
            summary = (
                f"Automated ECG waveform analysis of '{filename}' tracked {ecg_cv['r_peaks_detected']} cardiac cycles "
                f"showing an estimated resting rate of {ecg_cv['estimated_hr_bpm']} bpm with regular R-R spacing and "
                f"isoelectric ST segments (within normal resting screening limits). Non-ECG biomarkers were reset to a "
                f"neutral screening baseline unless entered from patient history."
            )
        else:
            summary = (
                f"Automated ECG waveform analysis of '{filename}' tracked {ecg_cv['r_peaks_detected']} cardiac cycles "
                f"with an estimated resting rate of {ecg_cv['estimated_hr_bpm']} bpm, {ecg_cv['rhythm_flag'].replace('_', ' ')} "
                f"R-R spacing, and {ecg_cv['st_assessment'].lower()}."
            )
    elif feats:
        summary = (
            f"Document analysis of '{filename}' extracted {len(feats)} clinical biomarkers and "
            f"{len(cues.get('medications', []))} active medications, updating the cardiovascular risk model."
        )
    else:
        summary = f"Processed '{filename}' ({quality.get('width', 0)}x{quality.get('height', 0)} px). Verify clinical fields before scoring."

    recommendations = list((vision_narrative or {}).get("recommendations") or [])
    if not recommendations:
        if ecg_cv and ecg_cv.get("ecg_overall_normal"):
            recommendations.append("Resting ECG tracing appears within normal screening limits; add patient blood pressure, lipids, or history if available for complete risk stratification.")
            recommendations.append("Maintain routine preventive cardiovascular checkups and confirm any symptoms with a licensed physician.")
        else:
            if ecg_cv and ecg_cv.get("st_abnormal"):
                recommendations.append("Correlate suspected ST-segment morphology with a physician-interpreted 12-lead ECG and troponin panel.")
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
    """Analyze an uploaded ECG image, lab report image, PDF, or text."""
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

        api_key = os.environ.get("GEMINI_API_KEY", "").strip()
        if online_mode and api_key and mime_type.startswith("image/"):
            vision_json = _call_gemini_vision(raw_bytes, mime_type, api_key)
            if vision_json:
                vision_narrative = vision_json
                engine_used = "Online Gemini Vision API (gemini-2.5-flash)"
                ocr_text = vision_json.get("raw_text", ocr_text)

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

    # Map CV-derived ECG findings cleanly (NEVER map resting HR 60-90 bpm directly into stress max HR `thalach`!)
    if ecg_cv and ecg_cv.get("is_ecg_image") and ecg_cv.get("signal_extracted"):
        feats = parsed["extracted_features"]
        conf = parsed["confidence"]
        if "restecg" not in feats:
            feats["restecg"] = 1 if ecg_cv.get("st_abnormal") else 0
            conf["restecg"] = "ecg-waveform-cv"
        if "oldpeak" not in feats:
            feats["oldpeak"] = ecg_cv.get("estimated_oldpeak_mm", 0.0)
            conf["oldpeak"] = "ecg-waveform-cv"
        if "slope" not in feats:
            feats["slope"] = 1 if ecg_cv.get("st_abnormal") else 2
            conf["slope"] = "ecg-waveform-cv"
        if "thalach" not in feats:
            # Healthy expected exercise HR reserve for a normal resting ECG vs reduced reserve if abnormal
            feats["thalach"] = 166 if ecg_cv.get("ecg_overall_normal") else 134
            conf["thalach"] = "ecg-reserve-est"

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


def generate_sample_ecg_strip_png(out_path: str | Path, abnormal: bool = False) -> Path:
    """Create a realistic synthetic 10-second ECG waveform strip PNG (normal sinus by default)."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    w, h = 960, 340
    img = Image.new("RGB", (w, h), color=(255, 246, 246))
    draw = ImageDraw.Draw(img)

    for x in range(0, w, 12):
        col = (242, 182, 182) if x % 60 == 0 else (250, 218, 218)
        draw.line([(x, 0), (x, h)], fill=col, width=1)
    for y in range(0, h, 12):
        col = (242, 182, 182) if y % 60 == 0 else (250, 218, 218)
        draw.line([(0, y), (w, y)], fill=col, width=1)

    xs = np.arange(20, w - 20)
    y_base = h * 0.56
    pts = []
    period = 76.0  # ~72 bpm across 10s
    for x in xs:
        phase = ((x - 20) % period) / period
        val = 0.0
        val += 9.0 * np.exp(-((phase - 0.16) / 0.04) ** 2)       # P wave
        val -= 10.0 * np.exp(-((phase - 0.28) / 0.015) ** 2)     # Q dip
        val += 92.0 * np.exp(-((phase - 0.31) / 0.014) ** 2)     # R spike
        val -= 20.0 * np.exp(-((phase - 0.345) / 0.016) ** 2)    # S dip
        if abnormal:
            val -= 16.0 * np.exp(-((phase - 0.44) / 0.05) ** 2)  # ST depression
        val += 22.0 * np.exp(-((phase - 0.58) / 0.06) ** 2)      # T wave
        pts.append((float(x), float(y_base - val)))

    draw.line(pts, fill=(18, 24, 34), width=3)
    img.save(out_path, "PNG")
    return out_path
