#!/usr/bin/env python3
"""Automated end-to-end verification tests for CardioPulse XAI v2.0."""
import base64
from pathlib import Path
from src.report_ocr import generate_sample_lab_report_png, analyze_report_payload
from src.patient_history import PatientHistoryDB, apply_history_modifier
from src.connectors import fetch_fhir_patient_bundle, fetch_openfda_drug_safety, fetch_pubmed_evidence

BASE = Path(__file__).resolve().parent

# 1. Generate & analyze sample medical lab report PNG
sample_png = generate_sample_lab_report_png(BASE / "assets" / "sample_lab_report.png")
b64 = base64.b64encode(sample_png.read_bytes()).decode("ascii")
ocr_res = analyze_report_payload(image_b64=b64, filename="sample_lab_report.png", online_mode=False)
feats = ocr_res["extracted_features"]
assert feats["age"] == 62, f"Expected age 62, got {feats}"
assert feats["trestbps"] == 152, f"Expected BP 152, got {feats}"
assert feats["chol"] == 286, f"Expected chol 286, got {feats}"
assert feats["fbs"] == 1, f"Expected fbs 1, got {feats}"
assert feats["oldpeak"] == 2.3, f"Expected oldpeak 2.3, got {feats}"
assert "atorvastatin" in ocr_res["history_cues"]["medications"]
print(f"[PASS] Image Report Analyzer extracted {len(feats)} biomarkers + {len(ocr_res['history_cues']['medications'])} meds via {ocr_res['engine']}")

# 2. Test SQLite Longitudinal Patient History & Bayesian Modifier
db = PatientHistoryDB(BASE / "data" / "cardiopulse_ehr.db")
patients = db.list_patients()
assert len(patients) >= 3
p = db.get_patient("CP-1042")
assert len(p["visits"]) >= 2
hist_eval = apply_history_modifier(0.76, p, p["visits"])
assert hist_eval["adjusted_risk"] > hist_eval["base_ml_risk"]
print(f"[PASS] Longitudinal EHR DB loaded {len(patients)} patients; CP-1042 base={hist_eval['base_ml_risk']} -> history-adjusted={hist_eval['adjusted_risk']}")

# 3. Test Graceful Offline Fallback for FHIR, openFDA, and PubMed Connectors
fhir_off = fetch_fhir_patient_bundle("example", online_mode=False)
assert fhir_off["status"] == "offline"
fda_off = fetch_openfda_drug_safety(["atorvastatin"], online_mode=False)
assert fda_off[0]["status"] == "offline"
pm_off = fetch_pubmed_evidence(["oldpeak", "exang"], online_mode=False)
assert pm_off[0]["status"] == "offline"
print("[PASS] Online Connectors (FHIR R4, openFDA, PubMed) verified with clean offline fallback")
