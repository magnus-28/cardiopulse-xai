#!/usr/bin/env python3
"""Automated end-to-end verification tests for CardioPulse XAI v2.0."""
import base64
from pathlib import Path
from src.report_ocr import (
    generate_sample_lab_report_png,
    generate_sample_ecg_strip_png,
    analyze_report_payload,
)
from src.patient_history import PatientHistoryDB, apply_history_modifier
from src.connectors import fetch_fhir_patient_bundle, fetch_openfda_drug_safety, fetch_pubmed_evidence

BASE = Path(__file__).resolve().parent

# 1. Test Lab Report PNG OCR + Detailed Report
sample_png = generate_sample_lab_report_png(BASE / "assets" / "sample_lab_report.png")
b64 = base64.b64encode(sample_png.read_bytes()).decode("ascii")
ocr_res = analyze_report_payload(image_b64=b64, filename="sample_lab_report.png", online_mode=False)
feats = ocr_res["extracted_features"]
assert feats["age"] == 62 and feats["trestbps"] == 152 and feats["chol"] == 286 and feats["fbs"] == 1
assert ocr_res["structured_report"]["executive_summary"]
print(f"[PASS] Lab Report Analyzer: {len(feats)} biomarkers + structured report via {ocr_res['engine']}")

# 2. Test Pure ECG Waveform Strip PNG (Zero printed text!) -> Waveform CV + Detailed Report
ecg_png = generate_sample_ecg_strip_png(BASE / "assets" / "sample_ecg_strip.png")
ecg_b64 = base64.b64encode(ecg_png.read_bytes()).decode("ascii")
ecg_res = analyze_report_payload(image_b64=ecg_b64, filename="test2.png", online_mode=False)
sig = ecg_res["structured_report"]["signal_metrics"]
assert sig is not None and sig["r_peaks_detected"] >= 5, f"Expected ECG R-peaks, got {ecg_res['structured_report']}"
assert ecg_res["extracted_features"].get("thalach") is not None
assert len(ecg_res["structured_report"]["findings"]) >= 3
print(f"[PASS] ECG Waveform Analyzer: HR={sig['estimated_hr_bpm']} bpm, peaks={sig['r_peaks_detected']}, ST_abn={sig['st_abnormal']}, mapped={ecg_res['extracted_features']}")

# 3. Test SQLite Longitudinal EHR DB
db = PatientHistoryDB(BASE / "data" / "cardiopulse_ehr.db")
p = db.get_patient("CP-1042")
hist_eval = apply_history_modifier(0.76, p, p["visits"])
assert hist_eval["adjusted_risk"] > hist_eval["base_ml_risk"]
print(f"[PASS] Longitudinal EHR DB: CP-1042 base={hist_eval['base_ml_risk']} -> adjusted={hist_eval['adjusted_risk']}")

# 4. Test Offline Connectors Fallback
assert fetch_fhir_patient_bundle("example", online_mode=False)["status"] == "offline"
assert fetch_openfda_drug_safety(["atorvastatin"], online_mode=False)[0]["status"] == "offline"
assert fetch_pubmed_evidence(["oldpeak"], online_mode=False)[0]["status"] == "offline"
print("[PASS] Hybrid Online Connectors verified with clean offline fallback")
