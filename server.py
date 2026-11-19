#!/usr/bin/env python3
"""
CardioPulse XAI — Interactive Multimodal Clinical Intelligence Server.
Runs seamlessly both locally (`python server.py` -> http://127.0.0.1:8000)
and online in the cloud (Render, Railway, Hugging Face Spaces, Docker -> 0.0.0.0:$PORT).
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
<title>CardioPulse XAI — Interactive Cardiac Intelligence</title>
<style>
  :root {
    --brand:#0A5C6B; --brand-dark:#073E49; --accent:#0EA5A4; --accent-soft:#E6F7F7;
    --coral:#EA580C; --danger:#DC2626; --warn:#D97706; --ok:#059669;
    --bg:#F4F8FB; --surface:#FFFFFF; --text:#0F172A; --muted:#52637A; --border:#E2EAF0;
    --shadow:0 4px 20px rgba(10, 62, 73, 0.06);
  }
  * { box-sizing:border-box; }
  body { margin:0; font-family:'Inter', system-ui, -apple-system, Segoe UI, Roboto, sans-serif; background:var(--bg); color:var(--text); }

  /* Crisp White Top Navbar */
  .navbar {
    position:sticky; top:0; z-index:50; background:rgba(255,255,255,0.95); backdrop-filter:blur(10px);
    border-bottom:1px solid var(--border); padding:10px 18px;
    display:flex; align-items:center; justify-content:space-between; gap:16px; flex-wrap:wrap;
  }
  .brand { display:flex; align-items:center; gap:10px; text-decoration:none; color:var(--brand-dark); }
  .brand-logo {
    width:38px; height:38px; border-radius:10px; background:linear-gradient(135deg, var(--brand), var(--accent));
    display:grid; place-items:center; box-shadow:0 4px 12px rgba(14,165,164,0.28);
  }
  .brand-name { font-size:18px; font-weight:800; letter-spacing:-0.02em; line-height:1.1; }
  .brand-tag { font-size:11px; font-weight:600; color:var(--accent); letter-spacing:0.04em; text-transform:uppercase; }

  .nav-links { display:flex; gap:6px; background:#EFF5F8; padding:5px; border-radius:12px; flex-wrap:wrap; }
  .nav-btn {
    border:none; background:transparent; color:var(--muted); padding:7px 12px; border-radius:9px;
    font-size:13px; font-weight:600; cursor:pointer; transition:all 0.2s ease; display:flex; align-items:center; gap:6px;
  }
  .nav-btn:hover { color:var(--brand-dark); background:rgba(255,255,255,0.65); }
  .nav-btn.active { background:#FFFFFF; color:var(--brand-dark); box-shadow:0 2px 8px rgba(15,23,42,0.08); }
  .nav-count { background:var(--accent-soft); color:var(--brand); font-size:11px; font-weight:700; padding:1px 7px; border-radius:99px; }

  .cta-btn {
    background:linear-gradient(135deg, var(--brand), var(--accent)); color:#FFFFFF; border:none;
    padding:8px 14px; border-radius:10px; white-space:nowrap; font-size:13px; font-weight:700; cursor:pointer;
    box-shadow:0 4px 12px rgba(10,92,107,0.22); transition:transform 0.15s, box-shadow 0.15s;
  }
  .cta-btn:hover { transform:translateY(-1px); box-shadow:0 6px 16px rgba(10,92,107,0.3); }

  /* Compact Interactive Hero Strip */
  .hero {
    background:linear-gradient(125deg, #073E49 0%, #0A5C6B 55%, #0D7A8A 100%);
    color:#FFFFFF; padding:20px 28px;
  }
  .hero-inner {
    max-width:1320px; margin:0 auto; display:flex; align-items:center; justify-content:space-between; gap:20px; flex-wrap:wrap;
  }
  .hero h1 { margin:0; font-size:22px; font-weight:800; letter-spacing:-0.015em; }
  .hero p { margin:4px 0 0; font-size:13px; color:#CBE9EE; }
  .hero-stats { display:flex; gap:12px; flex-wrap:wrap; }
  .stat-pill {
    background:rgba(255,255,255,0.12); border:1px solid rgba(255,255,255,0.2);
    padding:8px 14px; border-radius:12px; min-width:118px;
  }
  .stat-pill .val { font-size:17px; font-weight:800; color:#FFFFFF; }
  .stat-pill .lbl { font-size:11px; color:#CBE9EE; }

  /* Animated ECG Line */
  .ecg-wave { width:160px; height:36px; opacity:0.85; }
  .ecg-path { stroke:#5EEAD4; stroke-width:2.2; fill:none; stroke-dasharray:260; stroke-dashoffset:260; animation:pulseDraw 2.6s linear infinite; }
  @keyframes pulseDraw { 0% { stroke-dashoffset:260; } 100% { stroke-dashoffset:0; } }

  /* Main Container & Views */
  .container { max-width:1320px; margin:22px auto; padding:0 22px 42px; }
  .view-section { display:none; animation:fadeIn 0.22s ease; }
  .view-section.active { display:block; }
  @keyframes fadeIn { from { opacity:0; transform:translateY(6px); } to { opacity:1; transform:translateY(0); } }

  .grid-2 { display:grid; grid-template-columns:1fr 1fr; gap:20px; align-items:stretch; }
  .col-stack { display:flex; flex-direction:column; gap:20px; }
  .card {
    background:var(--surface); border:1px solid var(--border); border-radius:16px;
    padding:20px 22px; box-shadow:var(--shadow); transition:box-shadow 0.2s;
  }
  .card:hover { box-shadow:0 8px 28px rgba(10,62,73,0.085); }
  .card-grow { flex:1; display:flex; flex-direction:column; }

  .card-hdr {
    display:flex; justify-content:space-between; align-items:center; gap:10px; flex-wrap:wrap;
    margin:0 0 14px; padding-bottom:10px; border-bottom:1px solid #EEF3F7;
  }
  .card-hdr h2 { margin:0; font-size:15.5px; font-weight:750; color:var(--brand-dark); }

  /* Drag-and-Drop Zone */
  .dropzone {
    border:2px dashed #93C5D1; border-radius:14px; padding:18px; text-align:center;
    background:linear-gradient(180deg, #F7FCFD 0%, #EEF7F9 100%); cursor:pointer;
    transition:all 0.2s ease; position:relative; overflow:hidden;
  }
  .dropzone.dragover { border-color:var(--accent); background:#E0F5F5; transform:scale(1.01); }
  .dropzone.scanning::after {
    content:""; position:absolute; inset:0; background:linear-gradient(90deg, transparent, rgba(14,165,164,0.22), transparent);
    animation:scanSweep 1.1s linear infinite;
  }
  @keyframes scanSweep { from { transform:translateX(-100%); } to { transform:translateX(100%); } }
  .drop-title { font-size:14.5px; font-weight:700; color:var(--brand-dark); margin-bottom:4px; }
  .drop-sub { font-size:12px; color:var(--muted); margin-bottom:12px; }
  .btn-group { display:flex; gap:8px; justify-content:center; flex-wrap:wrap; }

  .btn {
    padding:8px 14px; border:none; border-radius:9px; font-weight:700; font-size:12.5px;
    cursor:pointer; transition:all 0.15s ease; display:inline-flex; align-items:center; gap:5px;
  }
  .btn:hover { transform:translateY(-1px); filter:brightness(0.96); }
  .btn-brand { background:var(--brand); color:#FFFFFF; }
  .btn-coral { background:var(--coral); color:#FFFFFF; }
  .btn-soft { background:var(--accent-soft); color:var(--brand-dark); }
  .btn-outline { background:#FFFFFF; color:var(--brand-dark); border:1px solid var(--border); }
  .btn-danger { background:#FEE2E2; color:#B91C1C; padding:5px 10px; font-size:11.5px; }

  .preview-wrap { display:none; margin-top:12px; }
  .preview-wrap img { max-height:150px; max-width:100%; border-radius:10px; border:1px solid var(--border); box-shadow:0 3px 10px rgba(0,0,0,0.06); }

  /* Interactive Sliders & Inputs */
  .slider-grid { display:grid; grid-template-columns:1fr 1fr; gap:12px; margin-bottom:12px; }
  .slider-card { background:#F8FBFD; border:1px solid #E5EFF4; border-radius:10px; padding:10px 12px; }
  .slider-top { display:flex; justify-content:space-between; font-size:12px; font-weight:700; color:var(--brand-dark); margin-bottom:6px; }
  .slider-val { color:var(--accent); font-weight:800; }
  input[type="range"] { width:100%; accent-color:var(--accent); cursor:pointer; height:6px; }

  .field-grid { display:grid; grid-template-columns:repeat(3, 1fr); gap:10px; }
  label.fld { display:block; font-size:11.5px; font-weight:600; color:var(--muted); margin-bottom:4px; }
  input[type="text"], input[type="number"], select, textarea {
    width:100%; padding:8px 10px; border:1px solid #CBD5E1; border-radius:9px;
    font-size:13px; background:#FFFFFF; color:var(--text); font-family:inherit;
  }
  input:focus, select:focus, textarea:focus { outline:none; border-color:var(--accent); box-shadow:0 0 0 3px rgba(14,165,164,0.15); }

  /* Animated Radial Risk Gauge */
  .gauge-wrap { display:grid; grid-template-columns:190px 1fr; gap:18px; align-items:center; margin-bottom:14px; }
  .gauge-box { position:relative; width:180px; height:120px; margin:0 auto; }
  .gauge-svg { width:180px; height:120px; }
  .gauge-bg { fill:none; stroke:#E6EEF3; stroke-width:14; stroke-linecap:round; }
  .gauge-fg { fill:none; stroke:var(--ok); stroke-width:14; stroke-linecap:round; transition:stroke-dashoffset 0.6s ease, stroke 0.4s ease; }
  .gauge-text { position:absolute; bottom:8px; left:0; right:0; text-align:center; }
  .gauge-pct { font-size:30px; font-weight:800; color:var(--brand-dark); line-height:1; }
  .gauge-sub { font-size:11px; font-weight:700; color:var(--muted); margin-top:3px; text-transform:uppercase; }

  .risk-metrics { display:grid; grid-template-columns:1fr 1fr; gap:10px; }
  .mini-kpi { background:#F7FAFC; border:1px solid var(--border); border-radius:10px; padding:10px 12px; }
  .mini-kpi .k-lbl { font-size:11px; font-weight:700; color:var(--muted); text-transform:uppercase; }
  .mini-kpi .k-val { font-size:20px; font-weight:800; color:var(--brand-dark); margin-top:2px; }
  .badge { display:inline-block; padding:4px 12px; border-radius:999px; font-size:12px; font-weight:700; color:#FFFFFF; }

  /* SHAP & What-If */
  .bar-row { display:grid; grid-template-columns:135px 1fr 56px; align-items:center; gap:8px; margin-bottom:8px; font-size:12.5px; }
  .bar-track { background:#E6EEF3; height:10px; border-radius:99px; overflow:hidden; }
  .bar-fill { height:100%; border-radius:99px; transition:width 0.45s ease; }
  .pill-box { border-left:4px solid var(--accent); background:#F0FDFA; padding:9px 12px; border-radius:8px; margin-bottom:8px; font-size:12.5px; }
  .warn-box { border-left:4px solid var(--coral); background:#FFF7ED; padding:9px 12px; border-radius:8px; margin-bottom:8px; font-size:12.5px; }

  /* Detailed Report Box */
  .report-box {
    background:#F8FBFD; border:1px solid #D5E7EE; border-left:4px solid var(--brand);
    border-radius:12px; padding:15px 17px; font-size:13px; line-height:1.55; flex:1;
  }
  .report-box ul { margin:6px 0 10px; padding-left:18px; }
  .chip-row { display:flex; flex-wrap:wrap; gap:6px; margin-bottom:8px; }
  .chip { background:var(--accent-soft); color:var(--brand-dark); font-size:11.5px; font-weight:700; padding:3px 10px; border-radius:99px; }

  /* Patient Directory & Saved Report Cards */
  .patient-list { display:grid; gap:10px; max-height:460px; overflow-y:auto; padding-right:4px; }
  .pat-card {
    border:1px solid var(--border); border-radius:12px; padding:12px 14px; background:#FAFCFD;
    cursor:pointer; transition:all 0.15s; display:flex; justify-content:space-between; align-items:center; gap:10px;
  }
  .pat-card:hover, .pat-card.selected { border-color:var(--accent); background:#EEF9F9; }
  .gallery-grid { display:grid; grid-template-columns:repeat(auto-fill, minmax(310px, 1fr)); gap:16px; }
  .saved-card {
    background:#FFFFFF; border:1px solid var(--border); border-radius:14px; padding:16px;
    box-shadow:var(--shadow); display:flex; flex-direction:column; gap:10px;
  }
  .saved-thumb { width:100%; height:130px; object-fit:cover; border-radius:10px; border:1px solid var(--border); background:#F1F5F9; }

  .chk-grid { display:flex; flex-wrap:wrap; gap:12px; margin:10px 0; font-size:12.5px; }
  .chk-grid label { display:flex; align-items:center; gap:5px; cursor:pointer; }
  .chk-grid input { width:auto; }
  .meta { font-size:12px; color:var(--muted); }

  /* Floating Toast */
  .toast {
    position:fixed; bottom:24px; right:24px; background:var(--brand-dark); color:#FFFFFF;
    padding:12px 18px; border-radius:12px; font-size:13px; font-weight:600;
    box-shadow:0 8px 24px rgba(7,62,73,0.28); opacity:0; transform:translateY(16px);
    transition:all 0.25s ease; pointer-events:none; z-index:99;
  }
  .toast.show { opacity:1; transform:translateY(0); }

  @media (max-width:960px) {
    .grid-2, .gauge-wrap, .slider-grid, .field-grid { grid-template-columns:1fr; }
  }
</style>
</head>
<body>

<!-- TOP WHITE NAVIGATION BAR -->
<nav class="navbar">
  <div class="brand">
    <div class="brand-logo">
      <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round">
        <path d="M22 12h-4l-3 9L9 3l-3 9H2"/>
      </svg>
    </div>
    <div>
      <div class="brand-name">CardioPulse XAI</div>
      <div class="brand-tag">Cardiac Intelligence Studio</div>
    </div>
  </div>

  <div class="nav-links">
    <button class="nav-btn active" onclick="openView('view-studio', this)">Diagnostics Studio</button>
    <button class="nav-btn" onclick="openView('view-patients', this)">Patients &amp; EHR <span class="nav-count" id="navPatCount">0</span></button>
    <button class="nav-btn" onclick="openView('view-reports', this)">Saved Reports <span class="nav-count" id="navRepCount">0</span></button>
    <button class="nav-btn" onclick="openView('view-intel', this)">Clinical Intel &amp; FHIR</button>
  </div>

  <button class="cta-btn" onclick="resetForNewScan()">+ New Assessment</button>
</nav>

<!-- INTERACTIVE HERO STRIP -->
<section class="hero">
  <div class="hero-inner">
    <div style="display:flex;align-items:center;gap:18px;flex-wrap:wrap;">
      <div>
        <h1>Explainable ECG &amp; Cardiovascular Risk Platform</h1>
        <p>Drag-and-drop ECG strips or lab reports, explore live What-If sliders, and archive longitudinal patient records.</p>
      </div>
      <svg class="ecg-wave" viewBox="0 0 200 40">
        <path class="ecg-path" d="M0,22 L38,22 L46,16 L52,22 L60,22 L66,30 L74,4 L82,34 L88,22 L106,22 L116,14 L126,22 L156,22 L164,16 L170,22 L200,22"/>
      </svg>
    </div>
    <div class="hero-stats">
      <div class="stat-pill"><div class="val" id="heroActivePat">Standalone</div><div class="lbl">Active Patient</div></div>
      <div class="stat-pill"><div class="val">88.4%</div><div class="lbl">Screening Sensitivity</div></div>
      <div class="stat-pill"><div class="val" id="heroLatency">-- ms</div><div class="lbl">Compute Time</div></div>
    </div>
  </div>
</section>

<div class="container">

  <!-- =================================================================== -->
  <!-- VIEW 1: DIAGNOSTICS STUDIO                                          -->
  <!-- =================================================================== -->
  <section id="view-studio" class="view-section active">
    <div class="grid-2">
      <!-- LEFT COLUMN -->
      <div class="col-stack">
        <div class="card">
          <div class="card-hdr">
            <h2>1. ECG &amp; Lab Report Upload</h2>
            <span class="meta" id="uploadHint">PNG · JPG · PDF</span>
          </div>
          <div id="dropzone" class="dropzone" onclick="document.getElementById('reportFile').click()">
            <div class="drop-title">Drop an ECG strip or lab report here, or click to browse</div>
            <div class="drop-sub">Digitizes 12-lead &amp; single-lead ECG waveforms and extracts lab biomarkers automatically</div>
            <input type="file" id="reportFile" accept="image/*,.pdf" style="display:none;" onchange="analyzeUploadedReport()"/>
            <div class="btn-group" onclick="event.stopPropagation()">
              <button class="btn btn-brand" onclick="document.getElementById('reportFile').click()">Upload Image / PDF</button>
              <button class="btn btn-soft" onclick="loadSampleECG()">Try Normal ECG</button>
              <button class="btn btn-soft" onclick="loadSampleReport()">Try Lab Report</button>
            </div>
            <div id="previewWrap" class="preview-wrap"><img id="previewImg" alt="Uploaded report preview"/></div>
          </div>
          <label style="display:flex;align-items:center;gap:7px;font-size:12px;color:var(--muted);margin-top:10px;cursor:pointer;">
            <input type="checkbox" id="keepPatientHistory" style="width:auto;margin:0;" onchange="runEvaluation(false)"/>
            <span>Apply active patient's EHR history to uploaded report</span>
          </label>
          <div id="reportStatus" class="meta" style="margin-top:6px;"></div>
        </div>

        <div class="card card-grow">
          <div class="card-hdr">
            <h2>2. Interactive Biomarker Simulator</h2>
            <span class="meta">Slide to see live risk changes</span>
          </div>
          <!-- Live Interactive Sliders -->
          <div class="slider-grid">
            <div class="slider-card">
              <div class="slider-top"><span>Age (years)</span><span class="slider-val" id="val_age">45</span></div>
              <input id="age" type="range" min="29" max="77" value="45" oninput="syncSlider('age')"/>
            </div>
            <div class="slider-card">
              <div class="slider-top"><span>Resting BP (mmHg)</span><span class="slider-val" id="val_trestbps">120</span></div>
              <input id="trestbps" type="range" min="94" max="200" value="120" oninput="syncSlider('trestbps')"/>
            </div>
            <div class="slider-card">
              <div class="slider-top"><span>Cholesterol (mg/dL)</span><span class="slider-val" id="val_chol">190</span></div>
              <input id="chol" type="range" min="126" max="420" value="190" oninput="syncSlider('chol')"/>
            </div>
            <div class="slider-card">
              <div class="slider-top"><span>Max Heart Rate (bpm)</span><span class="slider-val" id="val_thalach">166</span></div>
              <input id="thalach" type="range" min="71" max="202" value="166" oninput="syncSlider('thalach')"/>
            </div>
            <div class="slider-card" style="grid-column:1 / -1;">
              <div class="slider-top"><span>ST Depression / oldpeak (mm)</span><span class="slider-val" id="val_oldpeak">0.0</span></div>
              <input id="oldpeak" type="range" min="0" max="6.2" step="0.1" value="0.0" oninput="syncSlider('oldpeak')"/>
            </div>
          </div>

          <!-- Categorical Clinical Inputs -->
          <div class="field-grid">
            <div><label class="fld">Sex</label><select id="sex" onchange="debouncedEval()"><option value="0" selected>Female</option><option value="1">Male</option></select></div>
            <div><label class="fld">Chest Pain</label><select id="cp" onchange="debouncedEval()"><option value="0">Typical Angina</option><option value="1">Atypical Angina</option><option value="2" selected>Non-Anginal</option><option value="3">Asymptomatic</option></select></div>
            <div><label class="fld">Fasting Sugar &gt;120</label><select id="fbs" onchange="debouncedEval()"><option value="0" selected>No</option><option value="1">Yes</option></select></div>
            <div><label class="fld">Resting ECG</label><select id="restecg" onchange="debouncedEval()"><option value="0" selected>Normal</option><option value="1">ST-T Abnormality</option><option value="2">LV Hypertrophy</option></select></div>
            <div><label class="fld">Exercise Angina</label><select id="exang" onchange="debouncedEval()"><option value="0" selected>No</option><option value="1">Yes</option></select></div>
            <div><label class="fld">ST Slope</label><select id="slope" onchange="debouncedEval()"><option value="0">Downsloping</option><option value="1">Flat</option><option value="2" selected>Upsloping</option></select></div>
            <div><label class="fld">Major Vessels (0-3)</label><select id="ca" onchange="debouncedEval()"><option value="0" selected>0</option><option value="1">1</option><option value="2">2</option><option value="3">3</option></select></div>
            <div style="grid-column:span 2;"><label class="fld">Thalassemia Perfusion</label><select id="thal" onchange="debouncedEval()"><option value="1" selected>Normal (1)</option><option value="2">Fixed Defect (2)</option><option value="3">Reversible Defect (3)</option></select></div>
          </div>
          <div style="display:grid;grid-template-columns:1.2fr 1fr;gap:10px;margin-top:auto;padding-top:14px;">
            <button class="btn btn-coral" style="justify-content:center;" onclick="runEvaluation(false)">Refresh Risk &amp; XAI</button>
            <button class="btn btn-brand" style="justify-content:center;" onclick="saveCurrentReport()">Save Report to Patient</button>
          </div>
        </div>
      </div>

      <!-- RIGHT COLUMN -->
      <div class="col-stack">
        <div class="card">
          <div class="card-hdr">
            <h2>Live Risk Stratification &amp; Explainability</h2>
            <span id="riskBadge" class="badge" style="background:var(--ok);">Low Risk</span>
          </div>
          <div class="gauge-wrap">
            <div class="gauge-box">
              <svg class="gauge-svg" viewBox="0 0 180 115">
                <path class="gauge-bg" d="M 20 100 A 70 70 0 0 1 160 100"/>
                <path id="gaugeArc" class="gauge-fg" d="M 20 100 A 70 70 0 0 1 160 100" stroke-dasharray="220" stroke-dashoffset="200"/>
              </svg>
              <div class="gauge-text">
                <div class="gauge-pct" id="adjRisk">--%</div>
                <div class="gauge-sub">Composite Risk</div>
              </div>
            </div>
            <div class="risk-metrics">
              <div class="mini-kpi"><div class="k-lbl">Base ML Risk</div><div class="k-val" id="baseRisk">--%</div></div>
              <div class="mini-kpi"><div class="k-lbl">Trajectory</div><div class="k-val" id="trendText" style="font-size:14px;">Standalone</div></div>
            </div>
          </div>
          <div id="historyFactors"></div>

          <div style="font-size:13px;font-weight:700;color:var(--brand-dark);margin:10px 0 8px;">Top Local SHAP Drivers</div>
          <div id="shapBars"></div>

          <div style="font-size:13px;font-weight:700;color:var(--brand-dark);margin:12px 0 8px;">Actionable What-If Reductions</div>
          <div id="cfList"></div>
        </div>

        <div class="card card-grow">
          <div class="card-hdr">
            <h2>Detailed Diagnostic Report</h2>
            <div style="display:flex;gap:6px;">
              <button class="btn btn-brand" style="padding:5px 10px;font-size:11.5px;" onclick="saveCurrentReport()">Save</button>
              <button class="btn btn-soft" style="padding:5px 10px;font-size:11.5px;" onclick="downloadReportHtml()">Export HTML</button>
              <button class="btn btn-outline" style="padding:5px 10px;font-size:11.5px;" onclick="window.print()">Print</button>
            </div>
          </div>
          <div id="detailedReport" class="report-box">
            <strong>Ready for Diagnostics:</strong> Drop an ECG image or lab report on the left (or click <em>Try Normal ECG</em> / <em>Try Lab Report</em>) to generate a complete clinical summary with rhythm, rate, ST-segment findings, and follow-up guidance.
          </div>
        </div>
      </div>
    </div>
  </section>

  <!-- =================================================================== -->
  <!-- VIEW 2: PATIENTS & EHR                                              -->
  <!-- =================================================================== -->
  <section id="view-patients" class="view-section">
    <div class="grid-2">
      <div class="card">
        <div class="card-hdr">
          <h2>Patient Directory (SQLite EHR)</h2>
          <button class="btn btn-brand" onclick="startNewPatientForm()">+ Add Patient</button>
        </div>
        <div id="patientDirectory" class="patient-list"></div>
      </div>

      <div class="card">
        <div class="card-hdr">
          <h2>Patient Profile &amp; Medical History</h2>
          <button class="btn btn-soft" onclick="loadPatientIntoStudio()">Open in Studio →</button>
        </div>
        <div class="grid-2" style="gap:10px;">
          <div><label class="fld">Patient MRN / ID</label><input id="pat_mrn" type="text" placeholder="CP-1201"/></div>
          <div><label class="fld">Full Name</label><input id="pat_name" type="text" placeholder="Full Name"/></div>
          <div><label class="fld">Phone / Contact</label><input id="pat_phone" type="text" placeholder="+91-98XXX-XXXXX"/></div>
          <div><label class="fld">Blood Group</label>
            <select id="pat_blood"><option value="">Not specified</option><option>A+</option><option>A-</option><option>B+</option><option>B-</option><option>AB+</option><option>AB-</option><option>O+</option><option>O-</option></select>
          </div>
        </div>
        <div style="margin-top:10px;">
          <label class="fld">Active Medications (comma-separated)</label>
          <input id="medications" type="text" placeholder="atorvastatin, metformin, amlodipine"/>
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
          <textarea id="pat_notes" rows="3" placeholder="Symptoms, allergies, or relevant history..."></textarea>
        </div>
        <div id="visitTimeline" class="meta" style="margin:10px 0;"></div>
        <button class="btn btn-brand" style="width:100%;justify-content:center;" onclick="savePatientProfile()">Save Patient Details</button>
      </div>
    </div>
  </section>

  <!-- =================================================================== -->
  <!-- VIEW 3: SAVED REPORTS GALLERY                                       -->
  <!-- =================================================================== -->
  <section id="view-reports" class="view-section">
    <div class="card">
      <div class="card-hdr">
        <h2>Saved Analyzed Reports Archive</h2>
        <span class="meta">All ECG &amp; lab reports saved with their uploaded images</span>
      </div>
      <div id="savedGallery" class="gallery-grid"></div>
    </div>
  </section>

  <!-- =================================================================== -->
  <!-- VIEW 4: CLINICAL INTEL & FHIR                                       -->
  <!-- =================================================================== -->
  <section id="view-intel" class="view-section">
    <div class="grid-2">
      <div class="card card-grow">
        <div class="card-hdr"><h2>Medication Safety Screening (openFDA)</h2></div>
        <div id="fdaList" class="meta">Add patient medications or run an evaluation to check openFDA cardiovascular warnings.</div>
      </div>
      <div class="card card-grow">
        <div class="card-hdr"><h2>Peer-Reviewed Cardiology Evidence (PubMed)</h2></div>
        <div id="pubmedList" class="meta">Run an evaluation to retrieve PubMed citations matched to the patient's top SHAP drivers.</div>
      </div>
    </div>
    <div class="card" style="margin-top:20px;">
      <div class="card-hdr"><h2>Connect External Hospital EHR (HL7 FHIR R4)</h2></div>
      <div style="display:grid;grid-template-columns:1.5fr 1fr auto;gap:10px;align-items:end;">
        <div><label class="fld">FHIR R4 Base URL</label><input id="fhirUrl" type="text" value="https://hapi.fhir.org/baseR4"/></div>
        <div><label class="fld">FHIR Patient ID</label><input id="fhirPatientId" type="text" placeholder="e.g. example"/></div>
        <button class="btn btn-brand" onclick="pullFhirPatient()">Pull Record</button>
      </div>
      <div id="fhirStatus" class="meta" style="margin-top:8px;">Automatically falls back gracefully when offline.</div>
    </div>
  </section>

</div>

<div id="toast" class="toast"></div>

<script>
const FIELDS = ["age","sex","cp","trestbps","chol","fbs","restecg","thalach","exang","oldpeak","slope","ca","thal"];
const SLIDERS = ["age","trestbps","chol","thalach","oldpeak"];
let PATIENTS = [];
let ALL_REPORTS = [];
let ACTIVE_MRN = null;
let LAST_REPORT_DATA = null;
let LAST_EVAL_DATA = null;
let LAST_UPLOADED_IMAGE = "";
let evalTimer = null;

function showToast(msg) {
  const t = document.getElementById("toast");
  t.textContent = msg;
  t.classList.add("show");
  setTimeout(() => t.classList.remove("show"), 3200);
}

function openView(viewId, btn) {
  document.querySelectorAll(".view-section").forEach(v => v.classList.remove("active"));
  document.querySelectorAll(".nav-btn").forEach(b => b.classList.remove("active"));
  document.getElementById(viewId).classList.add("active");
  if (btn) btn.classList.add("active");
  if (viewId === "view-reports") fetchAllReports();
}

function syncSlider(id) {
  const v = document.getElementById(id).value;
  document.getElementById("val_" + id).textContent = id === "oldpeak" ? parseFloat(v).toFixed(1) : v;
  debouncedEval();
}

function setField(id, val) {
  const el = document.getElementById(id);
  if (!el) return;
  el.value = val;
  if (SLIDERS.includes(id)) {
    document.getElementById("val_" + id).textContent = id === "oldpeak" ? parseFloat(val).toFixed(1) : val;
  }
}

function debouncedEval() {
  clearTimeout(evalTimer);
  evalTimer = setTimeout(() => runEvaluation(false), 160);
}

function resetForNewScan() {
  const NEUTRAL = { age:45, sex:0, cp:2, trestbps:120, chol:190, fbs:0, restecg:0, thalach:166, exang:0, oldpeak:0.0, slope:2, ca:0, thal:1 };
  FIELDS.forEach(f => setField(f, NEUTRAL[f]));
  ["hist_smoking","hist_diabetes","hist_hypertension","hist_family","hist_mi"].forEach(id => { document.getElementById(id).checked = false; });
  document.getElementById("medications").value = "";
  document.getElementById("keepPatientHistory").checked = false;
  document.getElementById("previewWrap").style.display = "none";
  LAST_UPLOADED_IMAGE = "";
  LAST_REPORT_DATA = null;
  document.getElementById("heroActivePat").textContent = "New Screening";
  openView("view-studio", document.querySelector(".nav-btn"));
  runEvaluation(false);
  showToast("Reset to clean screening baseline.");
}

// Drag & Drop handlers
const dz = document.getElementById("dropzone");
["dragenter","dragover"].forEach(ev => dz.addEventListener(ev, e => { e.preventDefault(); dz.classList.add("dragover"); }));
["dragleave","drop"].forEach(ev => dz.addEventListener(ev, e => { e.preventDefault(); dz.classList.remove("dragover"); }));
dz.addEventListener("drop", e => {
  const f = e.dataTransfer.files[0];
  if (f) {
    const reader = new FileReader();
    reader.onload = () => analyzeReportPayload(reader.result, f.name);
    reader.readAsDataURL(f);
  }
});

async function fetchPatients(selectMrn = null) {
  const res = await fetch("/api/patients");
  PATIENTS = await res.json();
  document.getElementById("navPatCount").textContent = PATIENTS.length;
  if (selectMrn) ACTIVE_MRN = selectMrn;
  else if (!ACTIVE_MRN && PATIENTS.length) ACTIVE_MRN = PATIENTS[0].mrn;

  const dir = document.getElementById("patientDirectory");
  dir.innerHTML = PATIENTS.map(p => `
    <div class="pat-card ${p.mrn === ACTIVE_MRN ? 'selected' : ''}" onclick="selectPatient('${p.mrn}')">
      <div>
        <div style="font-weight:700;color:var(--brand-dark);">${p.full_name}</div>
        <div class="meta">${p.mrn} ${p.blood_group ? '· ' + p.blood_group : ''} · ${(p.visits||[]).length} visits · ${(p.saved_reports||[]).length} saved reports</div>
      </div>
      <span class="chip">Select</span>
    </div>
  `).join("");
  if (ACTIVE_MRN) populatePatientForm(ACTIVE_MRN);
  await fetchAllReports();
}

function selectPatient(mrn) {
  ACTIVE_MRN = mrn;
  populatePatientForm(mrn);
  document.querySelectorAll(".pat-card").forEach(c => c.classList.remove("selected"));
  fetchPatients(mrn);
}

function populatePatientForm(mrn) {
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
  tl.innerHTML = (p.visits && p.visits.length)
    ? "<strong>Visit Trajectory:</strong> " + p.visits.map(v => `${v.visit_date}: ${(v.adjusted_risk*100).toFixed(1)}%`).join(" → ")
    : "No recorded visits yet.";
}

function loadPatientIntoStudio() {
  const p = PATIENTS.find(x => x.mrn === ACTIVE_MRN);
  if (!p) return;
  document.getElementById("keepPatientHistory").checked = true;
  document.getElementById("heroActivePat").textContent = p.mrn;
  if (p.visits && p.visits.length) {
    const lf = p.visits[p.visits.length - 1].features;
    FIELDS.forEach(f => { if (lf[f] !== undefined) setField(f, lf[f]); });
  }
  openView("view-studio", document.querySelector(".nav-btn"));
  runEvaluation(false);
  showToast(`Loaded ${p.full_name} (${p.mrn}) into Diagnostics Studio.`);
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
  document.getElementById("pat_name").focus();
}

async function savePatientProfile() {
  const mrn = document.getElementById("pat_mrn").value.trim();
  const full_name = document.getElementById("pat_name").value.trim();
  if (!mrn || !full_name) { showToast("Enter both Patient MRN and Full Name."); return; }
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
  await fetchPatients(saved.mrn);
  showToast(`Saved patient ${saved.full_name} (${saved.mrn}).`);
}

async function fetchAllReports() {
  const res = await fetch("/api/reports");
  ALL_REPORTS = await res.json();
  document.getElementById("navRepCount").textContent = ALL_REPORTS.length;
  const gal = document.getElementById("savedGallery");
  if (!ALL_REPORTS.length) {
    gal.innerHTML = `<div class="meta">No saved reports yet. Analyze an ECG or lab image and click <strong>Save Report to Patient</strong>.</div>`;
    return;
  }
  gal.innerHTML = ALL_REPORTS.map(r => {
    const thumb = (r.image_data && r.image_data.startsWith("data:image/"))
      ? `<img class="saved-thumb" src="${r.image_data}" alt="${r.filename}"/>`
      : `<div class="saved-thumb" style="display:grid;place-items:center;color:var(--muted);font-size:12px;">Clinical Risk Record</div>`;
    const bcol = r.tier.includes("High") ? "#DC2626" : (r.tier.includes("Moderate") ? "#D97706" : "#059669");
    return `
      <div class="saved-card">
        ${thumb}
        <div style="display:flex;justify-content:space-between;align-items:center;">
          <strong style="color:var(--brand-dark);">${r.patient_name}</strong>
          <span class="badge" style="background:${bcol};">${(r.adjusted_risk*100).toFixed(1)}% · ${r.tier}</span>
        </div>
        <div class="meta">${r.mrn} · ${r.report_type} · ${r.created_at}</div>
        <div style="display:flex;gap:8px;margin-top:auto;">
          <button class="btn btn-soft" style="flex:1;justify-content:center;" onclick="viewSavedReport(${r.id})">Open Report</button>
          <button class="btn btn-danger" onclick="deleteSavedReport(${r.id})">Delete</button>
        </div>
      </div>`;
  }).join("");
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
  FIELDS.forEach(f => { if (r.features && r.features[f] !== undefined) setField(f, r.features[f]); });
  renderDetailedReport({
    structured_report: r.structured_report,
    image_data: LAST_UPLOADED_IMAGE,
    patient_banner: `${r.patient_name} (${r.mrn}) · Archived ${r.created_at}`
  });
  document.getElementById("heroActivePat").textContent = r.mrn;
  openView("view-studio", document.querySelector(".nav-btn"));
  await runEvaluation(false);
  showToast(`Opened archived report #${r.id} for ${r.patient_name}.`);
}

async function deleteSavedReport(reportId) {
  await fetch(`/api/reports/${reportId}`, { method:"DELETE" });
  await fetchPatients(ACTIVE_MRN);
  showToast("Deleted saved report.");
}

async function saveCurrentReport() {
  const mrn = document.getElementById("pat_mrn").value.trim() || ACTIVE_MRN || "CP-1042";
  const fullName = document.getElementById("pat_name").value.trim() || "Patient " + mrn;
  if (!LAST_EVAL_DATA) await runEvaluation(false);
  const features = {};
  FIELDS.forEach(f => features[f] = parseFloat(document.getElementById(f).value));
  const fallbackReport = {
    report_type: "Clinical Risk Assessment",
    engine: "CardioPulse XAI Ensemble",
    extracted_count: FIELDS.length,
    executive_summary: `Risk stratification for ${fullName} (${mrn}): Composite Risk ${(LAST_EVAL_DATA.history_eval.adjusted_risk*100).toFixed(1)}% (${LAST_EVAL_DATA.tier}).`,
    findings: (LAST_EVAL_DATA.shap || []).slice(0, 4).map(s => `SHAP Driver — ${s.feature}=${s.value}: ${s.shap >= 0 ? "+" : ""}${(s.shap*100).toFixed(1)} pts`),
    recommendations: (LAST_EVAL_DATA.counterfactuals || []).map(c => `${c.intervention} (-${(c.risk_reduction*100).toFixed(1)} pts)`),
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
  await fetchPatients(mrn);
  showToast(`Saved report #${saved.id} (with image) under ${saved.patient_name}.`);
}

function downloadReportHtml() {
  const mrn = document.getElementById("pat_mrn").value.trim() || "Patient";
  const name = document.getElementById("pat_name").value.trim() || "Screening";
  const baseR = document.getElementById("baseRisk").textContent;
  const adjR = document.getElementById("adjRisk").textContent;
  const tier = document.getElementById("riskBadge").textContent;
  const bodyHtml = document.getElementById("detailedReport").innerHTML;
  const doc = `<!DOCTYPE html><html><head><meta charset="utf-8"><title>CardioPulse Report - ${mrn}</title>
  <style>body{font-family:system-ui,sans-serif;max-width:820px;margin:32px auto;color:#0F172A;line-height:1.55;}
  h1{color:#0A5C6B;border-bottom:2px solid #0A5C6B;padding-bottom:8px;} .box{background:#F8FBFD;border:1px solid #CBD5E1;border-left:4px solid #0A5C6B;padding:16px;border-radius:10px;}</style></head>
  <body><h1>CardioPulse XAI — Clinical Diagnostic Report</h1>
  <p><strong>Patient:</strong> ${name} (${mrn}) | <strong>Base ML Risk:</strong> ${baseR} | <strong>Composite Risk:</strong> ${adjR} (${tier})</p>
  <div class="box">${bodyHtml}</div></body></html>`;
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([doc], { type:"text/html" }));
  a.download = `CardioPulse_Report_${mrn}.html`;
  a.click();
}

function renderDetailedReport(data) {
  const rep = data.structured_report;
  if (!rep) return;
  const chips = [
    `<span class="chip">${rep.report_type}</span>`,
    `<span class="chip">${rep.extracted_count} Biomarkers</span>`,
  ];
  if (rep.signal_metrics && rep.signal_metrics.estimated_hr_bpm) {
    chips.push(`<span class="chip">Resting HR: ${rep.signal_metrics.estimated_hr_bpm} bpm</span>`);
  }
  const banner = data.patient_banner ? `<div style="font-size:12px;font-weight:700;color:var(--brand);margin-bottom:6px;">${data.patient_banner}</div>` : "";
  const imgHtml = (data.image_data && data.image_data.startsWith("data:image/"))
    ? `<div style="margin:8px 0 10px;text-align:center;"><img src="${data.image_data}" alt="Analyzed report" style="max-height:165px;max-width:100%;border-radius:10px;border:1px solid #CBD5E1;"/></div>`
    : "";
  document.getElementById("detailedReport").innerHTML = `
    ${banner}
    <div class="chip-row">${chips.join("")}</div>
    ${imgHtml}
    <p style="margin:6px 0 8px;"><strong>Executive Summary:</strong> ${rep.executive_summary}</p>
    <strong>Signal &amp; Biomarker Findings:</strong>
    <ul>${(rep.findings || []).map(f => `<li>${f}</li>`).join("")}</ul>
    <strong>Recommended Follow-Up:</strong>
    <ul>${(rep.recommendations || []).map(r => `<li>${r}</li>`).join("")}</ul>
  `;
}

async function analyzeReportPayload(image_b64, filename) {
  dz.classList.add("scanning");
  document.getElementById("reportStatus").textContent = `Analyzing ${filename}...`;
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
    body:JSON.stringify({ image_b64, filename, online_mode: true })
  });
  const data = await res.json();
  dz.classList.remove("scanning");
  LAST_REPORT_DATA = data;

  // Always start a newly uploaded ECG/lab image from a clean biomarker baseline so old invasive values (ca, thal, cp) don't contaminate the scan
  const NEUTRAL = { age:45, sex:0, cp:2, trestbps:120, chol:190, fbs:0, restecg:0, thalach:166, exang:0, oldpeak:0.0, slope:2, ca:0, thal:1 };
  FIELDS.forEach(f => setField(f, NEUTRAL[f]));
  if (!document.getElementById("keepPatientHistory").checked) {
    ["hist_smoking","hist_diabetes","hist_hypertension","hist_family","hist_mi"].forEach(id => { document.getElementById(id).checked = false; });
    document.getElementById("medications").value = "";
    document.getElementById("heroActivePat").textContent = "Standalone Scan";
  }
  let count = 0;
  for (const [k, v] of Object.entries(data.extracted_features || {})) {
    setField(k, v); count++;
  }
  const hc = data.history_cues || {};
  if (hc.smoking) document.getElementById("hist_smoking").checked = true;
  if (hc.diabetes) document.getElementById("hist_diabetes").checked = true;
  if (hc.hypertension) document.getElementById("hist_hypertension").checked = true;
  if (hc.family_history) document.getElementById("hist_family").checked = true;
  if (hc.medications && hc.medications.length) document.getElementById("medications").value = hc.medications.join(", ");

  document.getElementById("reportStatus").innerHTML = `<strong>${data.structured_report.report_type}:</strong> Analyzed <em>${filename}</em> (${count} fields updated).`;
  renderDetailedReport({ ...data, image_data: LAST_UPLOADED_IMAGE });
  await runEvaluation(false);
  showToast(`Analyzed ${filename} successfully.`);
}

function analyzeUploadedReport() {
  const f = document.getElementById("reportFile").files[0];
  if (!f) return;
  const reader = new FileReader();
  reader.onload = () => analyzeReportPayload(reader.result, f.name);
  reader.readAsDataURL(f);
}

async function loadSampleReport() {
  const res = await fetch("/api/sample_report");
  const d = await res.json();
  await analyzeReportPayload("data:image/png;base64," + d.image_b64, "sample_lab_report.png");
}

async function loadSampleECG() {
  const res = await fetch("/api/sample_ecg");
  const d = await res.json();
  await analyzeReportPayload("data:image/png;base64," + d.image_b64, "sample_ecg_strip.png");
}

async function pullFhirPatient() {
  const server = document.getElementById("fhirUrl").value;
  const patient_id = document.getElementById("fhirPatientId").value.trim() || "example";
  document.getElementById("fhirStatus").textContent = "Querying HL7 FHIR R4...";
  const res = await fetch("/api/connect/fhir", {
    method:"POST", headers:{"Content-Type":"application/json"},
    body:JSON.stringify({ server, patient_id, online_mode: true })
  });
  const data = await res.json();
  if (data.status === "connected") {
    for (const [k, v] of Object.entries(data.mapped_features || {})) setField(k, v);
    document.getElementById("fhirStatus").innerHTML = `<strong>Connected:</strong> Loaded ${data.full_name}.`;
    openView("view-studio", document.querySelector(".nav-btn"));
    runEvaluation(false);
    showToast(`Pulled FHIR patient ${data.full_name}.`);
  } else {
    document.getElementById("fhirStatus").textContent = data.message || "FHIR lookup unavailable.";
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
  const attachMrn = document.getElementById("keepPatientHistory").checked ? ACTIVE_MRN : null;
  const res = await fetch("/api/predict", {
    method:"POST", headers:{"Content-Type":"application/json"},
    body:JSON.stringify({ mrn: attachMrn, features, history, save_visit: saveVisit, online_mode: true })
  });
  const data = await res.json();
  LAST_EVAL_DATA = data;

  const baseP = data.history_eval.base_ml_risk;
  const adjP = data.history_eval.adjusted_risk;
  document.getElementById("baseRisk").textContent = (baseP * 100).toFixed(1) + "%";
  document.getElementById("adjRisk").textContent = (adjP * 100).toFixed(1) + "%";
  document.getElementById("heroLatency").textContent = data.elapsed_ms + " ms";

  // Update Animated SVG Radial Gauge (arc length = 220)
  const arc = document.getElementById("gaugeArc");
  const offset = 220 - Math.min(1, Math.max(0, adjP)) * 220;
  arc.style.strokeDashoffset = offset.toFixed(1);
  const col = data.tier.includes("High") ? "#DC2626" : (data.tier.includes("Moderate") ? "#D97706" : "#059669");
  arc.style.stroke = col;

  const badge = document.getElementById("riskBadge");
  badge.textContent = data.tier;
  badge.style.background = col;

  const td = data.history_eval.trend_vs_last_visit;
  document.getElementById("trendText").textContent = td !== null ? `${td >= 0 ? "+" : ""}${(td*100).toFixed(1)} pts` : "Standalone";

  const hf = data.history_eval.applied_factors || [];
  document.getElementById("historyFactors").innerHTML = hf.length
    ? `<div class="warn-box"><strong>History Modifiers:</strong> ${hf.map(x => `${x.factor} (${x.log_odds_shift})`).join(" · ")}</div>`
    : "";

  const maxAbs = Math.max(...data.shap.map(s => Math.abs(s.shap)), 0.05);
  document.getElementById("shapBars").innerHTML = data.shap.map(s => {
    const pct = Math.min(100, (Math.abs(s.shap) / maxAbs) * 100);
    const barCol = s.shap >= 0 ? "#EA580C" : "#0EA5A4";
    const sign = s.shap >= 0 ? "+" : "";
    return `<div class="bar-row"><div>${s.feature} (${s.value})</div><div class="bar-track"><div class="bar-fill" style="width:${pct}%;background:${barCol};"></div></div><div style="text-align:right;font-weight:700;color:${barCol};">${sign}${(s.shap*100).toFixed(1)}%</div></div>`;
  }).join("");

  document.getElementById("cfList").innerHTML = (data.counterfactuals || []).map(c =>
    `<div class="pill-box"><strong>${c.intervention}</strong><br/>Estimated Risk: ${(c.new_risk*100).toFixed(1)}% (<strong>-${(c.risk_reduction*100).toFixed(1)} pts</strong>)</div>`
  ).join("") || "<div class='meta'>Patient is already near optimal modifiable biomarker range.</div>";

  document.getElementById("fdaList").innerHTML = (data.openfda || []).map(d =>
    `<div class="warn-box"><strong>${d.drug.toUpperCase()}</strong><br/>${d.warning || d.summary}</div>`
  ).join("") || "<div class='meta'>No active medications listed for screening.</div>";

  document.getElementById("pubmedList").innerHTML = (data.pubmed || []).map(p =>
    `<div class="pill-box"><a href="${p.url}" target="_blank" style="color:var(--brand-dark);font-weight:700;">${p.title}</a><br/><span class="meta">${p.journal}</span></div>`
  ).join("");
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
            self._send_json({"status": "ok", "service": "CardioPulse XAI", "threshold": CLIN_THR})
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
                online_mode=bool(body.get("online_mode", True)),
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
            online_mode = bool(body.get("online_mode", True))

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
