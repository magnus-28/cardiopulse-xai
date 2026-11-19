#!/usr/bin/env python3
"""
CardioPulse XAI v2.0 — Hybrid Edge + Cloud Clinical Intelligence Server.
Runs seamlessly both:
  - Locally on standard CPU laptops/workstations (`python server.py` -> http://127.0.0.1:8000)
  - Online in the cloud (Render, Railway, Hugging Face Spaces, Docker -> binds 0.0.0.0:$PORT)

Endpoints:
  GET    /                    Interactive Multimodal Clinical Studio UI
  GET    /health              Cloud load-balancer healthcheck
  GET    /api/sample_report   Returns bundled synthetic lab report PNG (base64)
  GET    /api/sample_ecg      Returns bundled synthetic ECG strip PNG (base64)
  GET    /api/patients        List SQLite EHR patients, visits & saved reports
  POST   /api/patients        Create or update patient details & medical history
  GET    /api/reports         List saved analyzed reports (?mrn=optional)
  GET    /api/reports/<id>    Get single full saved report
  POST   /api/reports         Save an analyzed report + risk assessment under a patient
  DELETE /api/reports/<id>    Delete a saved report
  POST   /api/analyze_report  Analyze uploaded ECG/lab image or PDF -> detailed report + auto-fill
  POST   /api/predict         Calibrated ML risk + Longitudinal History + SHAP + What-If + Online APIs
  POST   /api/connect/fhir    Pull patient demographics & observations from HL7 FHIR R4 server
"""
from __future__ import annotations
import base64
import json
import os
import pickle
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs
import numpy as np

from src.data_pipeline import FEATURE_NAMES
from src.explainability import compute_shapley_local, generate_counterfactual_plan
from src.patient_history import PatientHistoryDB, apply_history_modifier
from src.report_ocr import (
    analyze_report_payload,
    generate_sample_lab_report_png,
    generate_sample_ecg_strip_png,
)
from src.connectors import fetch_fhir_patient_bundle, fetch_openfda_drug_safety, fetch_pubmed_evidence

BASE_DIR = Path(__file__).resolve().parent
BUNDLE_PATH = BASE_DIR / "outputs" / "model_bundle.pkl"
SAMPLE_REPORT_PATH = BASE_DIR / "assets" / "sample_lab_report.png"
SAMPLE_ECG_PATH = BASE_DIR / "assets" / "sample_ecg_strip.png"

if not BUNDLE_PATH.exists():
    raise FileNotFoundError("Run `python train_and_evaluate.py` first to generate outputs/model_bundle.pkl.")
if not SAMPLE_REPORT_PATH.exists():
    generate_sample_lab_report_png(SAMPLE_REPORT_PATH)
if not SAMPLE_ECG_PATH.exists():
    generate_sample_ecg_strip_png(SAMPLE_ECG_PATH)

with open(BUNDLE_PATH, "rb") as f:
    BUNDLE = pickle.load(f)

SCALER = BUNDLE["scaler"]
ENSEMBLE = BUNDLE["ensemble"]
BG_X = BUNDLE["background_X"]
CLIN_THR = BUNDLE["clinical_threshold"]
EHR_DB = PatientHistoryDB(BASE_DIR / "data" / "cardiopulse_ehr.db")

HTML_UI = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>CardioPulse XAI — Clinical Intelligence Platform</title>
<style>
  :root {
    --primary:#0B3D4C; --primary-light:#135D72; --teal:#0E9AA7; --coral:#D9531E;
    --bg:#F1F5F8; --card:#FFFFFF; --text:#0F172A; --muted:#475569; --border:#DCE4EC;
  }
  * { box-sizing:border-box; }
  body { margin:0; font-family: 'Inter', system-ui, -apple-system, Segoe UI, Roboto, sans-serif; background:var(--bg); color:var(--text); }

  header {
    background:linear-gradient(120deg, #072630 0%, #0B3D4C 60%, #115366 100%);
    color:#FFFFFF; padding:16px 28px; display:flex; justify-content:space-between;
    align-items:center; flex-wrap:wrap; gap:14px; border-bottom:3px solid var(--coral);
    box-shadow:0 4px 14px rgba(7,38,48,0.18);
  }
  .brand-title { margin:0; font-size:20px; font-weight:800; letter-spacing:-0.015em; color:#FFFFFF; }
  .brand-sub { font-size:12.5px; color:#CBE3EA; margin-top:3px; }
  .mode-pill {
    display:inline-flex; align-items:center; gap:10px;
    background:rgba(255,255,255,0.14); border:1px solid rgba(255,255,255,0.28);
    padding:8px 15px; border-radius:999px; font-size:13px; color:#FFFFFF !important; cursor:pointer;
  }
  .mode-pill span { color:#FFFFFF !important; }
  .mode-pill input { width:16px; height:16px; accent-color:#38BDF8; margin:0; cursor:pointer; }

  .container { max-width:1340px; margin:22px auto; padding:0 20px 36px; }
  .workspace-grid { display:grid; grid-template-columns: 1fr 1fr; gap:20px; align-items:stretch; }
  .col-stack { display:flex; flex-direction:column; gap:18px; }
  .card {
    background:var(--card); border:1px solid var(--border); border-radius:14px;
    padding:20px; box-shadow:0 2px 12px rgba(15,23,42,0.045);
  }
  .card-grow { flex:1; display:flex; flex-direction:column; }

  h2 {
    margin:0 0 14px; font-size:15.5px; font-weight:700; color:var(--primary);
    display:flex; justify-content:space-between; align-items:center; gap:8px; flex-wrap:wrap;
    padding-bottom:8px; border-bottom:1px solid #EEF2F6;
  }
  h3 { margin:14px 0 8px; font-size:13.5px; font-weight:700; color:var(--primary); }

  .tabs { display:flex; gap:8px; margin-bottom:14px; flex-wrap:wrap; }
  .tab-btn {
    border:1px solid var(--border); background:#F8FAFC; color:var(--muted);
    padding:7px 13px; border-radius:8px; font-size:12.5px; font-weight:600; cursor:pointer; transition:0.15s;
  }
  .tab-btn.active { background:var(--primary); color:#FFFFFF; border-color:var(--primary); }
  .tab-panel { display:none; }
  .tab-panel.active { display:block; }

  .grid2 { display:grid; grid-template-columns:1fr 1fr; gap:10px; }
  .grid3 { display:grid; grid-template-columns:repeat(3, 1fr); gap:10px; }
  label.fld { display:block; font-size:11.5px; font-weight:600; color:var(--muted); margin-bottom:4px; }
  input, select, textarea { width:100%; padding:8px 10px; border:1px solid #CBD5E1; border-radius:8px; font-size:13px; background:#FFFFFF; color:var(--text); font-family:inherit; }
  input:focus, select:focus, textarea:focus { outline:none; border-color:var(--teal); box-shadow:0 0 0 3px rgba(14,154,167,0.14); }
  .btn { padding:8px 13px; border:none; border-radius:8px; font-weight:700; font-size:12.5px; cursor:pointer; transition:0.15s; }
  .btn:hover { filter:brightness(0.95); }
  .btn-primary { background:var(--coral); color:#FFFFFF; }
  .btn-sec { background:var(--primary); color:#FFFFFF; }
  .btn-soft { background:#E6F3F6; color:var(--primary); }
  .btn-danger { background:#FEE2E2; color:#B91C1C; padding:4px 9px; font-size:11.5px; }
  .btn-row { display:grid; grid-template-columns:1.35fr 1fr; gap:10px; margin-top:14px; }

  .dropzone {
    border:2px dashed #94A3B8; border-radius:10px; padding:14px; text-align:center;
    background:#F8FAFC; margin-bottom:10px;
  }
  .preview-wrap { display:none; margin-top:10px; text-align:center; }
  .preview-wrap img { max-height:145px; max-width:100%; border-radius:8px; border:1px solid var(--border); }

  .risk-strip { display:grid; grid-template-columns:1fr 1fr; gap:12px; margin-bottom:12px; }
  .risk-box { padding:14px 16px; border-radius:10px; background:#F1F6F8; border:1px solid #DCE8EC; }
  .risk-box.adj { background:#FFF6F0; border-color:#FBD9C4; }
  .risk-label { font-size:11px; font-weight:700; letter-spacing:0.04em; color:var(--muted); text-transform:uppercase; }
  .risk-num { font-size:32px; font-weight:800; color:var(--primary); line-height:1.15; margin-top:3px; }
  .badge { display:inline-block; padding:4px 11px; border-radius:999px; font-size:11.5px; font-weight:700; color:#FFFFFF; }

  .report-box {
    background:#F8FBFD; border:1px solid #D3E4EB; border-left:4px solid var(--primary);
    border-radius:10px; padding:14px 16px; font-size:13px; line-height:1.5;
  }
  .report-box ul { margin:6px 0 10px; padding-left:18px; }
  .report-box li { margin-bottom:4px; }
  .chip-row { display:flex; flex-wrap:wrap; gap:6px; margin:8px 0; }
  .chip { background:#E6F3F6; color:var(--primary); font-size:11.5px; font-weight:700; padding:3px 9px; border-radius:999px; }

  .saved-item {
    display:flex; justify-content:space-between; align-items:center; gap:10px;
    padding:9px 12px; border:1px solid var(--border); border-radius:8px; background:#F8FAFC;
    margin-bottom:7px; font-size:12.5px;
  }
  .bar-row { display:grid; grid-template-columns: 140px 1fr 58px; align-items:center; gap:8px; margin-bottom:7px; font-size:12.5px; }
  .bar-track { background:#E2E8F0; height:10px; border-radius:6px; overflow:hidden; }
  .bar-fill { height:100%; border-radius:6px; }
  .pill-box { border-left:4px solid var(--teal); background:#F0FDFA; padding:9px 12px; border-radius:7px; margin-bottom:8px; font-size:12.5px; }
  .warn-box { border-left:4px solid var(--coral); background:#FFF7ED; padding:9px 12px; border-radius:7px; margin-bottom:8px; font-size:12.5px; }
  .meta { font-size:12px; color:var(--muted); margin-top:6px; }
  .chk-grid { display:flex; flex-wrap:wrap; gap:12px; margin:10px 0; font-size:12.5px; }
  .chk-grid label { display:flex; align-items:center; gap:5px; cursor:pointer; }
  .chk-grid input { width:auto; }

  .bottom-grid { display:grid; grid-template-columns:1fr 1fr; gap:20px; margin-top:20px; align-items:stretch; }
  footer { text-align:center; font-size:12px; color:var(--muted); margin-top:22px; padding-top:12px; border-top:1px solid var(--border); }
  @media (max-width:960px) { .workspace-grid, .bottom-grid { grid-template-columns:1fr; } }
</style>
</head>
<body>
<header>
  <div>
    <h1 class="brand-title">CardioPulse XAI — Clinical Intelligence Platform</h1>
    <div class="brand-sub">ECG Waveform &amp; Lab Report Analyzer · Patient EHR &amp; Saved Reports · Hybrid Edge &amp; Cloud APIs</div>
  </div>
  <label class="mode-pill">
    <input type="checkbox" id="onlineMode" checked/>
    <span><strong>Hybrid Online Mode</strong> (FHIR · openFDA · PubMed · Vision API)</span>
  </label>
</header>

<div class="container">
  <div class="workspace-grid">
    <!-- LEFT COLUMN -->
    <div class="col-stack">
      <section class="card">
        <div class="tabs">
          <button class="tab-btn active" onclick="switchTab('tab-report', this)">1. Upload ECG / Lab Report</button>
          <button class="tab-btn" onclick="switchTab('tab-history', this)">2. Patient Details &amp; History</button>
          <button class="tab-btn" onclick="switchTab('tab-saved', this)">3. Saved Reports (<span id="savedCountBadge">0</span>)</button>
          <button class="tab-btn" onclick="switchTab('tab-fhir', this)">4. Online EHR (FHIR)</button>
        </div>

        <!-- TAB 1: UPLOAD ECG / LAB REPORT -->
        <div id="tab-report" class="tab-panel active">
          <div class="dropzone">
            <div style="font-weight:700;color:var(--primary);margin-bottom:8px;">Upload ECG Strip Image, Lab Report Photo, or PDF</div>
            <input type="file" id="reportFile" accept="image/*,.pdf" style="max-width:310px;margin:0 auto 10px;display:block;"/>
            <div style="display:flex;gap:8px;justify-content:center;flex-wrap:wrap;">
              <button class="btn btn-sec" onclick="analyzeUploadedReport()">Analyze Uploaded Image</button>
              <button class="btn btn-soft" onclick="loadSampleECG()">Demo ECG Strip</button>
              <button class="btn btn-soft" onclick="loadSampleReport()">Demo Lab Report</button>
            </div>
            <div id="previewWrap" class="preview-wrap"><img id="previewImg" alt="Uploaded report preview"/></div>
          </div>
          <label style="display:flex;align-items:center;gap:6px;font-size:12px;color:var(--muted);margin:6px 0 4px;cursor:pointer;">
            <input type="checkbox" id="keepPatientHistory" style="width:auto;margin:0;"/>
            <span>Combine upload with selected Patient History (unchecked = clean baseline for new patient)</span>
          </label>
          <div id="reportStatus" class="meta">Upload an ECG image or lab report — waveform signals &amp; biomarkers will generate a structured report automatically.</div>
        </div>

        <!-- TAB 2: ADD / EDIT PATIENT DETAILS & HISTORY -->
        <div id="tab-history" class="tab-panel">
          <div style="display:flex;gap:8px;align-items:end;margin-bottom:10px;">
            <div style="flex:1;">
              <label class="fld">Select Existing Patient (SQLite EHR)</label>
              <select id="patientSelect" onchange="loadSelectedPatient()"></select>
            </div>
            <button class="btn btn-soft" onclick="startNewPatientForm()">+ New Patient</button>
          </div>
          <div class="grid2">
            <div><label class="fld">Patient MRN / ID</label><input id="pat_mrn" placeholder="e.g. CP-1201"/></div>
            <div><label class="fld">Full Name</label><input id="pat_name" placeholder="e.g. Aarav Sharma"/></div>
            <div><label class="fld">Phone / Contact</label><input id="pat_phone" placeholder="+91-98XXX-XXXXX"/></div>
            <div><label class="fld">Blood Group</label>
              <select id="pat_blood">
                <option value="">Not specified</option>
                <option>A+</option><option>A-</option><option>B+</option><option>B-</option>
                <option>AB+</option><option>AB-</option><option>O+</option><option>O-</option>
              </select>
            </div>
          </div>
          <div style="margin-top:8px;">
            <label class="fld">Active Medications (comma-separated)</label>
            <input id="medications" placeholder="atorvastatin, metformin, amlodipine"/>
          </div>
          <div class="chk-grid">
            <label><input type="checkbox" id="hist_smoking"/> Smoking</label>
            <label><input type="checkbox" id="hist_diabetes"/> Diabetes</label>
            <label><input type="checkbox" id="hist_hypertension"/> Hypertension</label>
            <label><input type="checkbox" id="hist_family"/> Family History CVD</label>
            <label><input type="checkbox" id="hist_mi"/> Prior MI / Stroke</label>
          </div>
          <div>
            <label class="fld">Clinical Notes</label>
            <textarea id="pat_notes" rows="2" placeholder="Symptoms, allergies, or relevant medical history..."></textarea>
          </div>
          <div style="display:flex;gap:8px;margin-top:10px;align-items:center;">
            <button class="btn btn-sec" onclick="savePatientProfile()">Save Patient Profile</button>
            <span id="patientSaveStatus" class="meta" style="margin:0;"></span>
          </div>
          <div id="visitTimeline" class="meta" style="margin-top:8px;"></div>
        </div>

        <!-- TAB 3: SAVED ANALYZED REPORTS ARCHIVE -->
        <div id="tab-saved" class="tab-panel">
          <div id="savedReportsList" class="meta">No saved reports yet for this patient. Run an analysis and click "Save Report" to archive it here.</div>
        </div>

        <!-- TAB 4: ONLINE FHIR CONNECTOR -->
        <div id="tab-fhir" class="tab-panel">
          <div class="grid2">
            <div><label class="fld">HL7 FHIR R4 Server URL</label><input id="fhirUrl" value="https://hapi.fhir.org/baseR4"/></div>
            <div><label class="fld">External FHIR Patient ID</label><input id="fhirPatientId" placeholder="e.g. example or 12345"/></div>
          </div>
          <button class="btn btn-sec" style="margin-top:10px;" onclick="pullFhirPatient()">Pull Patient from FHIR R4</button>
          <div id="fhirStatus" class="meta">Connects to HAPI FHIR R4 or any hospital EHR endpoint when Hybrid Online Mode is active.</div>
        </div>
      </section>

      <section class="card card-grow">
        <h2>Clinical Biomarkers (13 UCI Features) <span style="font-size:11.5px;font-weight:500;color:var(--muted);">Auto-filled or editable</span></h2>
        <div class="grid3">
          <div><label class="fld">Age (yrs)</label><input id="age" type="number" value="62"/></div>
          <div><label class="fld">Sex</label><select id="sex"><option value="1" selected>Male (1)</option><option value="0">Female (0)</option></select></div>
          <div><label class="fld">Chest Pain (0-3)</label><select id="cp"><option value="0" selected>Typical (0)</option><option value="1">Atypical (1)</option><option value="2">Non-Anginal (2)</option><option value="3">Asymptomatic (3)</option></select></div>
          <div><label class="fld">Resting BP (mmHg)</label><input id="trestbps" type="number" value="152"/></div>
          <div><label class="fld">Cholesterol (mg/dL)</label><input id="chol" type="number" value="286"/></div>
          <div><label class="fld">Fasting Sugar &gt;120</label><select id="fbs"><option value="0">No (0)</option><option value="1" selected>Yes (1)</option></select></div>
          <div><label class="fld">Resting ECG</label><select id="restecg"><option value="0">Normal (0)</option><option value="1" selected>ST-T Abn (1)</option><option value="2">LVH (2)</option></select></div>
          <div><label class="fld">Max Heart Rate</label><input id="thalach" type="number" value="124"/></div>
          <div><label class="fld">Exercise Angina</label><select id="exang"><option value="0">No (0)</option><option value="1" selected>Yes (1)</option></select></div>
          <div><label class="fld">ST Depression (mm)</label><input id="oldpeak" type="number" step="0.1" value="2.3"/></div>
          <div><label class="fld">ST Slope</label><select id="slope"><option value="0">Down (0)</option><option value="1" selected>Flat (1)</option><option value="2">Up (2)</option></select></div>
          <div><label class="fld">Major Vessels (0-3)</label><input id="ca" type="number" value="2" min="0" max="3"/></div>
          <div><label class="fld">Thalassemia</label><select id="thal"><option value="1">Normal (1)</option><option value="2">Fixed (2)</option><option value="3" selected>Reversible (3)</option></select></div>
        </div>
        <div class="btn-row" style="margin-top:auto;padding-top:14px;">
          <button class="btn btn-primary" onclick="runEvaluation(false)">Run Full Risk &amp; XAI Analysis</button>
          <button class="btn btn-sec" onclick="saveCurrentReport()">Save Analyzed Report</button>
        </div>
      </section>
    </div>

    <!-- RIGHT COLUMN -->
    <div class="col-stack">
      <section class="card">
        <h2>
          <span>Risk Stratification (ML Ensemble + History)</span>
          <span id="riskBadge" class="badge" style="background:#64748B;">Ready</span>
        </h2>
        <div class="risk-strip">
          <div class="risk-box">
            <div class="risk-label">Base ML Ensemble Risk</div>
            <div class="risk-num" id="baseRisk">--%</div>
          </div>
          <div class="risk-box adj">
            <div class="risk-label">History-Adjusted Risk</div>
            <div class="risk-num" id="adjRisk" style="color:var(--coral);">--%</div>
            <div class="meta" id="trendText" style="margin:2px 0 0;"></div>
          </div>
        </div>
        <div id="historyFactors"></div>

        <h3>Top Local SHAP Feature Drivers</h3>
        <div id="shapBars"></div>

        <h3>Actionable What-If Risk Reductions</h3>
        <div id="cfList"></div>
      </section>

      <section class="card card-grow">
        <h2>
          <span>Detailed Diagnostic Report</span>
          <div style="display:flex;gap:6px;">
            <button class="btn btn-sec" style="padding:5px 10px;font-size:11.5px;" onclick="saveCurrentReport()">Save to Patient</button>
            <button class="btn btn-soft" style="padding:5px 10px;font-size:11.5px;" onclick="downloadReportHtml()">Download (.html)</button>
            <button class="btn btn-soft" style="padding:5px 10px;font-size:11.5px;" onclick="window.print()">Print</button>
          </div>
        </h2>
        <div id="saveReportBanner" class="meta" style="margin:0 0 8px;color:#0F766E;font-weight:600;"></div>
        <div id="detailedReport" class="report-box" style="flex:1;">
          <strong>Awaiting Report Upload:</strong> Upload an ECG waveform strip or lab report image (or click <em>Demo ECG Strip</em> / <em>Demo Lab Report</em>) to generate a complete structured clinical report, then save it under any patient profile.
        </div>
        <div class="meta" id="latencyMeta"></div>
      </section>
    </div>
  </div>

  <!-- BOTTOM EQUAL-HEIGHT 2-COLUMN ENRICHMENT ROW -->
  <div class="bottom-grid">
    <section class="card card-grow">
      <h2>Medication Safety Screening (openFDA Drug Label API)</h2>
      <div id="fdaList" class="meta">Run evaluation to check active patient medications for cardiovascular warnings.</div>
    </section>
    <section class="card card-grow">
      <h2>Live Peer-Reviewed Evidence (NCBI PubMed E-utilities)</h2>
      <div id="pubmedList" class="meta">Run evaluation to retrieve cardiology literature matched to top SHAP drivers.</div>
    </section>
  </div>

  <footer>
    CardioPulse XAI · Clinical Decision-Support Research Prototype · Always confirm findings with a licensed physician.
  </footer>
</div>

<script>
const FIELDS = ["age","sex","cp","trestbps","chol","fbs","restecg","thalach","exang","oldpeak","slope","ca","thal"];
let PATIENTS = [];
let LAST_REPORT_DATA = null;
let LAST_EVAL_DATA = null;
let LAST_UPLOADED_IMAGE = '';

function switchTab(id, btn) {
  document.querySelectorAll(".tab-panel").forEach(p => p.classList.remove("active"));
  document.querySelectorAll(".tab-btn").forEach(b => b.classList.remove("active"));
  document.getElementById(id).classList.add("active");
  btn.classList.add("active");
}

async function fetchPatients(selectMrn = null) {
  const res = await fetch("/api/patients");
  PATIENTS = await res.json();
  const sel = document.getElementById("patientSelect");
  sel.innerHTML = PATIENTS.map(p => `<option value="${p.mrn}">${p.mrn} — ${p.full_name}</option>`).join("");
  if (selectMrn && PATIENTS.some(p => p.mrn === selectMrn)) {
    sel.value = selectMrn;
  }
  loadSelectedPatient();
}

function startNewPatientForm() {
  const randId = "CP-" + Math.floor(1200 + Math.random() * 8500);
  document.getElementById("pat_mrn").value = randId;
  document.getElementById("pat_name").value = "";
  document.getElementById("pat_phone").value = "";
  document.getElementById("pat_blood").value = "";
  document.getElementById("medications").value = "";
  document.getElementById("pat_notes").value = "";
  ["hist_smoking","hist_diabetes","hist_hypertension","hist_family","hist_mi"].forEach(id => { document.getElementById(id).checked = false; });
  document.getElementById("patientSaveStatus").textContent = "Fill details and click Save Patient Profile.";
  document.getElementById("pat_name").focus();
}

async function savePatientProfile() {
  const mrn = document.getElementById("pat_mrn").value.trim();
  const full_name = document.getElementById("pat_name").value.trim();
  if (!mrn || !full_name) {
    alert("Please enter both Patient MRN / ID and Full Name.");
    return;
  }
  const payload = {
    mrn, full_name,
    phone: document.getElementById("pat_phone").value.trim(),
    blood_group: document.getElementById("pat_blood").value,
    medications: document.getElementById("medications").value,
    smoking: document.getElementById("hist_smoking").checked,
    diabetes: document.getElementById("hist_diabetes").checked,
    hypertension: document.getElementById("hist_hypertension").checked,
    family_history: document.getElementById("hist_family").checked,
    prior_mi_stroke: document.getElementById("hist_mi").checked,
    notes: document.getElementById("pat_notes").value.trim(),
  };
  const res = await fetch("/api/patients", { method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify(payload) });
  const saved = await res.json();
  document.getElementById("patientSaveStatus").textContent = `Saved ${saved.mrn} (${saved.full_name}).`;
  await fetchPatients(saved.mrn);
}

function loadSelectedPatient() {
  const mrn = document.getElementById("patientSelect").value;
  const p = PATIENTS.find(x => x.mrn === mrn);
  if (!p) return;
  document.getElementById("pat_mrn").value = p.mrn;
  document.getElementById("pat_name").value = p.full_name || "";
  document.getElementById("pat_phone").value = p.phone || "";
  document.getElementById("pat_blood").value = p.blood_group || "";
  document.getElementById("pat_notes").value = p.notes || "";
  document.getElementById("hist_smoking").checked = !!p.smoking;
  document.getElementById("hist_diabetes").checked = !!p.diabetes;
  document.getElementById("hist_hypertension").checked = !!p.hypertension;
  document.getElementById("hist_family").checked = !!p.family_history;
  document.getElementById("hist_mi").checked = !!p.prior_mi_stroke;
  document.getElementById("medications").value = (p.medications || []).join(", ");

  const tl = document.getElementById("visitTimeline");
  if (p.visits && p.visits.length) {
    tl.innerHTML = "<strong>Prior Visits:</strong> " + p.visits.map(v => `${v.visit_date}: ${(v.adjusted_risk*100).toFixed(1)}% (${v.tier})`).join(" → ");
    const lastFeats = p.visits[p.visits.length - 1].features;
    FIELDS.forEach(f => { if (lastFeats[f] !== undefined) document.getElementById(f).value = lastFeats[f]; });
  } else {
    tl.textContent = "No prior visits recorded yet.";
  }

  // Render Saved Reports Archive for selected patient
  const reps = p.saved_reports || [];
  document.getElementById("savedCountBadge").textContent = reps.length;
  const listEl = document.getElementById("savedReportsList");
  if (!reps.length) {
    listEl.innerHTML = `<div class="meta">No saved reports yet for <strong>${p.full_name}</strong> (${p.mrn}). Click <em>Save Analyzed Report</em> to archive one here.</div>`;
  } else {
    listEl.innerHTML = reps.map(r => `
      <div class="saved-item">
        <div>
          <strong>${r.report_type}</strong> · <span style="color:var(--muted);">${r.created_at}</span><br/>
          <span style="color:var(--muted);">File: ${r.filename}${r.has_image ? " · 📷 Image saved" : ""} · Risk: <strong>${(r.adjusted_risk*100).toFixed(1)}%</strong> (${r.tier})</span>
        </div>
        <div style="display:flex;gap:6px;">
          <button class="btn btn-soft" style="padding:4px 9px;font-size:11.5px;" onclick="viewSavedReport(${r.id})">View</button>
          <button class="btn btn-danger" onclick="deleteSavedReport(${r.id})">Delete</button>
        </div>
      </div>
    `).join("");
  }
}

async function viewSavedReport(reportId) {
  const res = await fetch(`/api/reports/${reportId}`);
  const r = await res.json();
  if (!r || r.error) return;
  LAST_REPORT_DATA = { filename: r.filename, structured_report: r.structured_report };
  LAST_UPLOADED_IMAGE = r.image_data || "";
  if (LAST_UPLOADED_IMAGE && LAST_UPLOADED_IMAGE.startsWith("data:image/")) {
    document.getElementById("previewImg").src = LAST_UPLOADED_IMAGE;
    document.getElementById("previewWrap").style.display = "block";
  } else {
    document.getElementById("previewWrap").style.display = "none";
  }
  FIELDS.forEach(f => { if (r.features && r.features[f] !== undefined) document.getElementById(f).value = r.features[f]; });
  renderDetailedReport({
    structured_report: r.structured_report,
    image_data: LAST_UPLOADED_IMAGE,
    patient_banner: `${r.patient_name} (${r.mrn}) · Saved ${r.created_at}`
  });
  document.getElementById("saveReportBanner").textContent = `Viewing archived report #${r.id} for ${r.patient_name} (${r.mrn}).`;
  await runEvaluation(false);
}

async function deleteSavedReport(reportId) {
  await fetch(`/api/reports/${reportId}`, { method:"DELETE" });
  const curMrn = document.getElementById("patientSelect").value;
  await fetchPatients(curMrn);
}

async function saveCurrentReport() {
  const mrn = document.getElementById("pat_mrn").value.trim() || document.getElementById("patientSelect").value;
  const fullName = document.getElementById("pat_name").value.trim() || "New Patient";
  if (!mrn) { alert("Please select or add a patient in Tab 2 first."); return; }
  if (!LAST_EVAL_DATA) await runEvaluation(false);

  // Ensure patient details are saved if user edited them
  await fetch("/api/patients", {
    method:"POST", headers:{"Content-Type":"application/json"},
    body:JSON.stringify({
      mrn, full_name: fullName,
      phone: document.getElementById("pat_phone").value.trim(),
      blood_group: document.getElementById("pat_blood").value,
      medications: document.getElementById("medications").value,
      smoking: document.getElementById("hist_smoking").checked,
      diabetes: document.getElementById("hist_diabetes").checked,
      hypertension: document.getElementById("hist_hypertension").checked,
      family_history: document.getElementById("hist_family").checked,
      prior_mi_stroke: document.getElementById("hist_mi").checked,
      notes: document.getElementById("pat_notes").value.trim(),
    })
  });

  const features = {};
  FIELDS.forEach(f => features[f] = parseFloat(document.getElementById(f).value));
  const fallbackReport = {
    report_type: "Clinical Risk Assessment",
    engine: "CardioPulse XAI Calibrated Ensemble",
    extracted_count: FIELDS.length,
    executive_summary: `Risk stratification for ${fullName} (${mrn}): Base ML risk ${(LAST_EVAL_DATA.history_eval.base_ml_risk*100).toFixed(1)}%, History-Adjusted risk ${(LAST_EVAL_DATA.history_eval.adjusted_risk*100).toFixed(1)}% (${LAST_EVAL_DATA.tier}).`,
    findings: (LAST_EVAL_DATA.shap || []).slice(0, 4).map(s => `SHAP Driver — ${s.feature}=${s.value}: ${s.shap >= 0 ? "+" : ""}${(s.shap*100).toFixed(1)} pts`),
    recommendations: (LAST_EVAL_DATA.counterfactuals || []).map(c => `${c.intervention} (est. -${(c.risk_reduction*100).toFixed(1)} pts)`),
  };

  const payload = {
    mrn, patient_name: fullName,
    filename: (LAST_REPORT_DATA && LAST_REPORT_DATA.filename) || "clinical_assessment",
    base_ml_risk: LAST_EVAL_DATA.history_eval.base_ml_risk,
    adjusted_risk: LAST_EVAL_DATA.history_eval.adjusted_risk,
    tier: LAST_EVAL_DATA.tier,
    structured_report: (LAST_REPORT_DATA && LAST_REPORT_DATA.structured_report) || fallbackReport,
    features,
    shap: LAST_EVAL_DATA.shap || [],
    image_data: LAST_UPLOADED_IMAGE || "",
  };
  const res = await fetch("/api/reports", { method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify(payload) });
  const saved = await res.json();
  document.getElementById("saveReportBanner").textContent = `Saved Report #${saved.id} under ${saved.patient_name} (${saved.mrn}) at ${saved.created_at}.`;
  await fetchPatients(mrn);
}

function downloadReportHtml() {
  const mrn = document.getElementById("pat_mrn").value.trim() || "Patient";
  const name = document.getElementById("pat_name").value.trim() || "Unassigned";
  const baseR = document.getElementById("baseRisk").textContent;
  const adjR = document.getElementById("adjRisk").textContent;
  const tier = document.getElementById("riskBadge").textContent;
  const bodyHtml = document.getElementById("detailedReport").innerHTML;
  const doc = `<!DOCTYPE html><html><head><meta charset="utf-8"><title>CardioPulse XAI Report - ${mrn}</title>
  <style>body{font-family:system-ui,sans-serif;max-width:820px;margin:32px auto;color:#0F172A;line-height:1.55;}
  h1{color:#0B3D4C;border-bottom:2px solid #0B3D4C;padding-bottom:8px;} .meta{color:#475569;font-size:13px;margin-bottom:16px;}
  .box{background:#F8FBFD;border:1px solid #CBD5E1;border-left:4px solid #0B3D4C;padding:16px;border-radius:8px;}</style></head>
  <body><h1>CardioPulse XAI — Clinical Diagnostic Report</h1>
  <div class="meta"><strong>Patient:</strong> ${name} (${mrn}) &nbsp;|&nbsp; <strong>Base ML Risk:</strong> ${baseR} &nbsp;|&nbsp; <strong>History-Adjusted Risk:</strong> ${adjR} (${tier})</div>
  <div class="box">${bodyHtml}</div>
  <p style="font-size:12px;color:#64748B;margin-top:24px;">Generated by CardioPulse XAI (Decision-support prototype; verify with a licensed physician).</p></body></html>`;
  const blob = new Blob([doc], { type: "text/html" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = `CardioPulse_Report_${mrn}.html`;
  a.click();
}

function renderDetailedReport(data) {
  const rep = data.structured_report;
  if (!rep) return;
  const chips = [
    `<span class="chip">${rep.report_type}</span>`,
    `<span class="chip">Engine: ${rep.engine}</span>`,
    `<span class="chip">${rep.extracted_count} Biomarkers Mapped</span>`,
  ];
  if (rep.signal_metrics && rep.signal_metrics.estimated_hr_bpm) {
    chips.push(`<span class="chip">Est. HR: ${rep.signal_metrics.estimated_hr_bpm} bpm</span>`);
  }
  const banner = data.patient_banner ? `<div style="font-size:12px;font-weight:700;color:var(--primary);margin-bottom:6px;">${data.patient_banner}</div>` : "";
  const imgHtml = (data.image_data && data.image_data.startsWith("data:image/"))
    ? `<div style="margin:8px 0 10px;text-align:center;"><img src="${data.image_data}" alt="Saved report image" style="max-height:175px;max-width:100%;border-radius:8px;border:1px solid #CBD5E1;"/></div>`
    : "";
  const findingsHtml = (rep.findings || []).map(f => `<li>${f}</li>`).join("");
  const recsHtml = (rep.recommendations || []).map(r => `<li>${r}</li>`).join("");
  document.getElementById("detailedReport").innerHTML = `
    ${banner}
    <div class="chip-row">${chips.join("")}</div>
    ${imgHtml}
    <p style="margin:6px 0 8px;"><strong>Executive Clinical Summary:</strong> ${rep.executive_summary}</p>
    <strong>Key Signal &amp; Biomarker Findings:</strong>
    <ul>${findingsHtml}</ul>
    <strong>Recommended Follow-Up Actions:</strong>
    <ul>${recsHtml}</ul>
  `;
}

async function analyzeReportPayload(image_b64, filename) {
  document.getElementById("reportStatus").textContent = "Analyzing report image & waveform signal...";
  document.getElementById("saveReportBanner").textContent = "";
  if (image_b64 && image_b64.startsWith("data:image/")) {
    LAST_UPLOADED_IMAGE = image_b64;
    document.getElementById("previewImg").src = image_b64;
    document.getElementById("previewWrap").style.display = "block";
  } else {
    LAST_UPLOADED_IMAGE = "";
    document.getElementById("previewWrap").style.display = "none";
  }
  const res = await fetch("/api/analyze_report", {
    method:"POST", headers:{"Content-Type":"application/json"},
    body:JSON.stringify({ image_b64, filename, online_mode: document.getElementById("onlineMode").checked })
  });
  const data = await res.json();
  LAST_REPORT_DATA = data;
  const combineHistory = document.getElementById("keepPatientHistory").checked;
  if (!combineHistory) {
    const NEUTRAL = { age:45, sex:0, cp:2, trestbps:120, chol:190, fbs:0, restecg:0, thalach:166, exang:0, oldpeak:0.0, slope:2, ca:0, thal:1 };
    FIELDS.forEach(f => { document.getElementById(f).value = NEUTRAL[f]; });
    ["hist_smoking","hist_diabetes","hist_hypertension","hist_family","hist_mi"].forEach(id => { document.getElementById(id).checked = false; });
    document.getElementById("medications").value = "";
  }
  let count = 0;
  for (const [k, v] of Object.entries(data.extracted_features || {})) {
    const el = document.getElementById(k);
    if (el) { el.value = v; count++; }
  }
  const hc = data.history_cues || {};
  if (hc.smoking) document.getElementById("hist_smoking").checked = true;
  if (hc.diabetes) document.getElementById("hist_diabetes").checked = true;
  if (hc.hypertension) document.getElementById("hist_hypertension").checked = true;
  if (hc.family_history) document.getElementById("hist_family").checked = true;
  if (hc.medications && hc.medications.length) document.getElementById("medications").value = hc.medications.join(", ");

  document.getElementById("reportStatus").innerHTML = `<strong>${data.engine}:</strong> Generated detailed report for <em>${data.filename}</em> (${count} clinical inputs updated).`;
  renderDetailedReport({ ...data, image_data: LAST_UPLOADED_IMAGE });
  runEvaluation(false);
}

function analyzeUploadedReport() {
  const file = document.getElementById("reportFile").files[0];
  if (!file) { alert("Please select an ECG image, lab report photo, or PDF first."); return; }
  const reader = new FileReader();
  reader.onload = () => analyzeReportPayload(reader.result, file.name);
  reader.readAsDataURL(file);
}

async function loadSampleReport() {
  const res = await fetch("/api/sample_report");
  const data = await res.json();
  await analyzeReportPayload("data:image/png;base64," + data.image_b64, "sample_lab_report.png");
}

async function loadSampleECG() {
  const res = await fetch("/api/sample_ecg");
  const data = await res.json();
  await analyzeReportPayload("data:image/png;base64," + data.image_b64, "sample_ecg_strip.png");
}

async function pullFhirPatient() {
  const server = document.getElementById("fhirUrl").value;
  const patient_id = document.getElementById("fhirPatientId").value.trim() || "example";
  document.getElementById("fhirStatus").textContent = "Querying HL7 FHIR R4 server...";
  const res = await fetch("/api/connect/fhir", {
    method:"POST", headers:{"Content-Type":"application/json"},
    body:JSON.stringify({ server, patient_id, online_mode: document.getElementById("onlineMode").checked })
  });
  const data = await res.json();
  if (data.status === "connected") {
    for (const [k, v] of Object.entries(data.mapped_features || {})) {
      const el = document.getElementById(k); if (el) el.value = v;
    }
    document.getElementById("fhirStatus").innerHTML = `<strong>Connected:</strong> Loaded <em>${data.full_name}</em> from ${data.server}.`;
    runEvaluation(false);
  } else {
    document.getElementById("fhirStatus").textContent = data.message || "FHIR query failed.";
  }
}

async function runEvaluation(saveVisit) {
  const features = {};
  FIELDS.forEach(f => features[f] = parseFloat(document.getElementById(f).value));
  const history = {
    smoking: document.getElementById("hist_smoking").checked,
    diabetes: document.getElementById("hist_diabetes").checked,
    hypertension: document.getElementById("hist_hypertension").checked,
    family_history: document.getElementById("hist_family").checked,
    prior_mi_stroke: document.getElementById("hist_mi").checked,
    medications: document.getElementById("medications").value.split(",").map(s => s.trim().toLowerCase()).filter(Boolean),
  };
  const attachMrn = (document.getElementById("keepPatientHistory").checked || document.getElementById("tab-history").classList.contains("active"))
    ? (document.getElementById("patientSelect").value || null)
    : null;
  const payload = {
    mrn: attachMrn,
    features, history, save_visit: saveVisit,
    online_mode: document.getElementById("onlineMode").checked
  };
  const res = await fetch("/api/predict", { method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify(payload) });
  const data = await res.json();
  LAST_EVAL_DATA = data;

  document.getElementById("baseRisk").textContent = (data.history_eval.base_ml_risk * 100).toFixed(1) + "%";
  document.getElementById("adjRisk").textContent = (data.history_eval.adjusted_risk * 100).toFixed(1) + "%";
  const td = data.history_eval.trend_vs_last_visit;
  document.getElementById("trendText").textContent = td !== null ? `Trajectory vs last visit: ${td >= 0 ? "+" : ""}${(td*100).toFixed(1)} pts` : "Standalone / first visit";

  const badge = document.getElementById("riskBadge");
  badge.textContent = data.tier;
  badge.style.background = data.tier.includes("High") ? "#DC2626" : (data.tier.includes("Moderate") ? "#D97706" : "#0D9488");

  const hf = data.history_eval.applied_factors || [];
  document.getElementById("historyFactors").innerHTML = hf.length
    ? `<div class="warn-box"><strong>History Modifiers Applied:</strong> ${hf.map(x => `${x.factor} (${x.log_odds_shift})`).join(" · ")}</div>`
    : "";

  const shapDiv = document.getElementById("shapBars");
  const maxAbs = Math.max(...data.shap.map(s => Math.abs(s.shap)), 0.05);
  shapDiv.innerHTML = data.shap.map(s => {
    const pct = Math.min(100, (Math.abs(s.shap) / maxAbs) * 100);
    const col = s.shap >= 0 ? "#D9531E" : "#0E9AA7";
    const sign = s.shap >= 0 ? "+" : "";
    return `<div class="bar-row"><div>${s.feature} (${s.value})</div><div class="bar-track"><div class="bar-fill" style="width:${pct}%;background:${col};"></div></div><div style="text-align:right;font-weight:700;color:${col};">${sign}${(s.shap*100).toFixed(1)}%</div></div>`;
  }).join("");

  document.getElementById("cfList").innerHTML = (data.counterfactuals || []).map(c =>
    `<div class="pill-box"><strong>${c.intervention}</strong><br/>New Base Risk: ${(c.new_risk*100).toFixed(1)}% (<strong>-${(c.risk_reduction*100).toFixed(1)} pts</strong>)</div>`
  ).join("") || "<div class='meta'>No modifiable risk reductions above threshold.</div>";

  document.getElementById("fdaList").innerHTML = (data.openfda || []).map(d =>
    `<div class="warn-box"><strong>${d.drug.toUpperCase()}</strong> (${d.status})<br/>${d.warning || d.summary}</div>`
  ).join("") || "<div class='meta'>No active medications listed.</div>";

  document.getElementById("pubmedList").innerHTML = (data.pubmed || []).map(p =>
    `<div class="pill-box"><a href="${p.url}" target="_blank" style="color:var(--primary);font-weight:700;">${p.title}</a><br/><span class="meta">${p.journal}</span></div>`
  ).join("");

  document.getElementById("latencyMeta").textContent = `Total compute time: ${data.elapsed_ms} ms | Mode: ${data.online_mode ? "Hybrid Online" : "Offline Edge"}`;
  if (saveVisit) fetchPatients(attachMrn);
}

window.onload = async () => { await fetchPatients(); await runEvaluation(false); };
</script>
</body>
</html>
"""


class HybridHandler(BaseHTTPRequestHandler):
    def _send_json(self, obj, status=200):
        raw = json.dumps(obj).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        qs = parse_qs(parsed.query)

        if path == "/":
            raw = HTML_UI.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)
        elif path == "/health":
            self._send_json({"status": "ok", "service": "CardioPulse XAI v2.0", "threshold": CLIN_THR})
        elif path == "/api/patients":
            self._send_json(EHR_DB.list_patients())
        elif path == "/api/reports":
            mrn = (qs.get("mrn") or [None])[0]
            self._send_json(EHR_DB.list_reports(mrn=mrn))
        elif path.startswith("/api/reports/"):
            try:
                rid = int(path.rsplit("/", 1)[-1])
                rep = EHR_DB.get_report(rid)
                self._send_json(rep if rep else {"error": "Report not found"}, status=200 if rep else 404)
            except ValueError:
                self._send_json({"error": "Invalid report ID"}, status=400)
        elif path == "/api/sample_report":
            b64 = base64.b64encode(SAMPLE_REPORT_PATH.read_bytes()).decode("ascii")
            self._send_json({"filename": "sample_lab_report.png", "image_b64": b64})
        elif path == "/api/sample_ecg":
            b64 = base64.b64encode(SAMPLE_ECG_PATH.read_bytes()).decode("ascii")
            self._send_json({"filename": "sample_ecg_strip.png", "image_b64": b64})
        else:
            self._send_json({"error": "Not found"}, status=404)

    def do_DELETE(self):
        path = urlparse(self.path).path
        if path.startswith("/api/reports/"):
            try:
                rid = int(path.rsplit("/", 1)[-1])
                ok = EHR_DB.delete_report(rid)
                self._send_json({"deleted": ok, "id": rid}, status=200 if ok else 404)
            except ValueError:
                self._send_json({"error": "Invalid report ID"}, status=400)
        else:
            self._send_json({"error": "Not found"}, status=404)

    def do_POST(self):
        path = urlparse(self.path).path
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length).decode("utf-8")) if length > 0 else {}

        if path == "/api/analyze_report":
            result = analyze_report_payload(
                image_b64=body.get("image_b64"),
                raw_text=body.get("raw_text"),
                filename=body.get("filename", "report.png"),
                online_mode=bool(body.get("online_mode", False)),
            )
            self._send_json(result)
            return

        if path == "/api/patients":
            saved = EHR_DB.upsert_patient(body)
            self._send_json(saved)
            return

        if path == "/api/reports":
            try:
                saved = EHR_DB.save_report(body)
                self._send_json(saved)
            except ValueError as e:
                self._send_json({"error": str(e)}, status=400)
            return

        if path == "/api/connect/fhir":
            res = fetch_fhir_patient_bundle(
                patient_id=str(body.get("patient_id", "example")),
                fhir_base_url=body.get("server"),
                online_mode=bool(body.get("online_mode", True)),
            )
            self._send_json(res)
            return

        if path == "/api/predict":
            t0 = time.perf_counter()
            feats = body.get("features", {})
            history = body.get("history", {})
            mrn = body.get("mrn")
            online_mode = bool(body.get("online_mode", False))

            x_raw = np.array([[float(feats[f]) for f in FEATURE_NAMES]], dtype=float)
            x_std = SCALER.transform(x_raw)
            base_risk = float(ENSEMBLE.predict_proba(x_std)[0])

            patient_rec = EHR_DB.get_patient(mrn) if mrn else None
            prior_visits = patient_rec["visits"] if patient_rec else []
            hist_eval = apply_history_modifier(base_risk, history, prior_visits)
            adj_risk = hist_eval["adjusted_risk"]
            tier = "High Risk" if adj_risk >= CLIN_THR else ("Moderate Risk" if adj_risk >= 0.25 else "Low Risk")

            _, sv = compute_shapley_local(ENSEMBLE.predict_proba, x_std[0], BG_X, n_permutations=40, seed=7)
            top_shap = sorted(
                [{"feature": f, "value": feats[f], "shap": float(s)} for f, s in zip(FEATURE_NAMES, sv)],
                key=lambda r: abs(r["shap"]), reverse=True,
            )[:6]
            cfs = generate_counterfactual_plan(lambda arr: ENSEMBLE.predict_proba(SCALER.transform(arr)), feats)[:3]

            if body.get("save_visit") and mrn:
                EHR_DB.record_visit(mrn, feats, base_risk, adj_risk, tier)

            meds = history.get("medications", [])
            openfda = fetch_openfda_drug_safety(meds, online_mode=online_mode)
            pubmed = fetch_pubmed_evidence([s["feature"] for s in top_shap if s["shap"] > 0], online_mode=online_mode)

            self._send_json({
                "tier": tier,
                "threshold": CLIN_THR,
                "history_eval": hist_eval,
                "shap": top_shap,
                "counterfactuals": cfs,
                "openfda": openfda,
                "pubmed": pubmed,
                "online_mode": online_mode,
                "elapsed_ms": round((time.perf_counter() - t0) * 1000.0, 1),
            })
            return

        self._send_json({"error": "Not found"}, status=404)

    def log_message(self, format, *args):
        return


def run_server():
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8000"))
    server = ThreadingHTTPServer((host, port), HybridHandler)
    print(f"CardioPulse XAI running on http://{host}:{port}  (Local browser: http://127.0.0.1:{port})")
    server.serve_forever()


if __name__ == "__main__":
    run_server()
