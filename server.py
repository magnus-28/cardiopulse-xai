#!/usr/bin/env python3
"""
CardioPulse XAI v2.0 — Hybrid Edge + Cloud Clinical Intelligence Server.
Runs BOTH:
  - Locally on Lenovo IdeaPad Slim 5i Ultra: `python server.py` -> http://127.0.0.1:8000
  - Online on Render / Railway / Hugging Face Spaces / Docker: binds 0.0.0.0:$PORT

Endpoints:
  GET  /                    Interactive Multimodal Clinical Studio UI
  GET  /health              Cloud load-balancer healthcheck
  GET  /api/sample_report   Returns bundled synthetic lab report PNG (base64) for 1-click demo
  GET  /api/patients        List SQLite EHR patients & longitudinal visits
  POST /api/patients        Create/update patient medical history
  POST /api/analyze_report  Analyze uploaded report image/PDF/text -> auto-fill clinical form
  POST /api/predict         Calibrated ML risk + Longitudinal History modifier + SHAP + What-If + Optional Online Enrichment
  POST /api/connect/fhir    Pull patient demographics & observations from HL7 FHIR R4 server
"""
from __future__ import annotations
import base64
import json
import os
import pickle
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import numpy as np

from src.data_pipeline import FEATURE_NAMES
from src.explainability import compute_shapley_local, generate_counterfactual_plan
from src.patient_history import PatientHistoryDB, apply_history_modifier
from src.report_ocr import analyze_report_payload, generate_sample_lab_report_png
from src.connectors import fetch_fhir_patient_bundle, fetch_openfda_drug_safety, fetch_pubmed_evidence

BASE_DIR = Path(__file__).resolve().parent
BUNDLE_PATH = BASE_DIR / "outputs" / "model_bundle.pkl"
SAMPLE_REPORT_PATH = BASE_DIR / "assets" / "sample_lab_report.png"

if not BUNDLE_PATH.exists():
    raise FileNotFoundError("Run `python train_and_evaluate.py` first to generate outputs/model_bundle.pkl.")
if not SAMPLE_REPORT_PATH.exists():
    generate_sample_lab_report_png(SAMPLE_REPORT_PATH)

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
<title>CardioPulse XAI v2.0 — Hybrid Multimodal Clinical Studio</title>
<style>
  :root { --primary:#0F4C5C; --teal:#1B9AAA; --coral:#E36414; --bg:#F4F7F9; --card:#FFFFFF; --text:#0F172A; --muted:#475569; --border:#DCE4EA; }
  * { box-sizing:border-box; }
  body { margin:0; font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; background:var(--bg); color:var(--text); }
  header { background:linear-gradient(120deg,#08242C,#0F4C5C); color:white; padding:16px 26px; display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:12px; }
  header h1 { margin:0; font-size:20px; letter-spacing:-0.01em; }
  .mode-pill { display:flex; align-items:center; gap:10px; background:rgba(255,255,255,0.12); padding:8px 14px; border-radius:999px; font-size:13px; }
  .mode-pill input { width:auto; margin:0; }
  main { max-width:1280px; margin:20px auto; padding:0 18px; display:grid; grid-template-columns: 1.05fr 1.15fr; gap:20px; }
  .card { background:var(--card); border:1px solid var(--border); border-radius:12px; padding:18px; box-shadow:0 2px 10px rgba(15,23,42,0.04); margin-bottom:16px; }
  h2 { margin:0 0 12px; font-size:15.5px; color:var(--primary); display:flex; justify-content:space-between; align-items:center; }
  .tabs { display:flex; gap:8px; margin-bottom:14px; flex-wrap:wrap; }
  .tab-btn { border:1px solid var(--border); background:#F8FAFC; color:var(--text); padding:7px 12px; border-radius:8px; font-size:13px; font-weight:600; cursor:pointer; }
  .tab-btn.active { background:var(--primary); color:white; border-color:var(--primary); }
  .tab-panel { display:none; }
  .tab-panel.active { display:block; }
  .grid2 { display:grid; grid-template-columns:1fr 1fr; gap:10px; }
  .grid3 { display:grid; grid-template-columns:1fr 1fr 1fr; gap:10px; }
  label { display:block; font-size:12px; font-weight:600; color:var(--muted); margin-bottom:3px; }
  input, select, textarea { width:100%; padding:7px 10px; border:1px solid #CBD5E1; border-radius:8px; font-size:13.5px; }
  .btn { padding:10px 14px; border:none; border-radius:8px; font-weight:700; font-size:13.5px; cursor:pointer; }
  .btn-primary { background:var(--coral); color:white; width:100%; margin-top:12px; }
  .btn-sec { background:var(--primary); color:white; }
  .btn-soft { background:#E6F2F4; color:var(--primary); }
  .dropzone { border:2px dashed #94A3B8; border-radius:10px; padding:16px; text-align:center; background:#F8FAFC; margin-bottom:10px; }
  .risk-strip { display:grid; grid-template-columns:1fr 1fr; gap:12px; margin-bottom:14px; }
  .risk-box { padding:14px; border-radius:10px; background:#F1F5F9; }
  .risk-num { font-size:32px; font-weight:800; color:var(--primary); }
  .badge { display:inline-block; padding:4px 10px; border-radius:999px; font-size:12px; font-weight:700; color:white; }
  .bar-row { display:grid; grid-template-columns: 145px 1fr 60px; align-items:center; gap:8px; margin-bottom:7px; font-size:12.5px; }
  .bar-track { background:#E2E8F0; height:11px; border-radius:6px; overflow:hidden; }
  .bar-fill { height:100%; border-radius:6px; }
  .pill-box { border-left:4px solid var(--teal); background:#F0FDFA; padding:9px 12px; border-radius:6px; margin-bottom:8px; font-size:12.5px; }
  .warn-box { border-left:4px solid var(--coral); background:#FFF7ED; padding:9px 12px; border-radius:6px; margin-bottom:8px; font-size:12.5px; }
  .meta { font-size:12px; color:var(--muted); margin-top:6px; }
  .chk-grid { display:flex; flex-wrap:wrap; gap:12px; margin:8px 0; font-size:13px; }
  .chk-grid label { display:flex; align-items:center; gap:5px; margin:0; color:var(--text); }
  .chk-grid input { width:auto; }
  @media (max-width:920px) { main { grid-template-columns:1fr; } }
</style>
</head>
<body>
<header>
  <div>
    <h1>CardioPulse XAI v2.0 — Hybrid Multimodal Clinical Studio</h1>
    <div style="font-size:12.5px;opacity:0.85;">Image Report Analyzer · Longitudinal Patient History · Edge + Online FHIR / openFDA / PubMed</div>
  </div>
  <label class="mode-pill">
    <input type="checkbox" id="onlineMode" checked/>
    <span><strong>Hybrid Online Mode</strong> (FHIR · openFDA · PubMed · Vision API)</span>
  </label>
</header>

<main>
  <div>
    <section class="card">
      <div class="tabs">
        <button class="tab-btn active" onclick="switchTab('tab-report', this)">1. Upload Report Image</button>
        <button class="tab-btn" onclick="switchTab('tab-history', this)">2. Patient History (EHR)</button>
        <button class="tab-btn" onclick="switchTab('tab-fhir', this)">3. Connect Online EHR (FHIR)</button>
      </div>

      <!-- TAB 1: IMAGE REPORT ANALYZER -->
      <div id="tab-report" class="tab-panel active">
        <div class="dropzone">
          <div style="font-weight:700;color:var(--primary);margin-bottom:6px;">Upload Lab Report or ECG Image (PNG / JPG / PDF)</div>
          <input type="file" id="reportFile" accept="image/*,.pdf" style="max-width:300px;margin:0 auto 10px;display:block;"/>
          <div style="display:flex;gap:8px;justify-content:center;flex-wrap:wrap;">
            <button class="btn btn-sec" onclick="analyzeUploadedReport()">Analyze Report Image</button>
            <button class="btn btn-soft" onclick="loadSampleReport()">Load Demo Lab Report PNG</button>
          </div>
        </div>
        <div id="reportStatus" class="meta">Upload a medical report image or click "Load Demo Lab Report PNG" to auto-extract biomarkers &amp; medications.</div>
      </div>

      <!-- TAB 2: LONGITUDINAL PATIENT HISTORY -->
      <div id="tab-history" class="tab-panel">
        <div class="grid2">
          <div>
            <label>Select Patient Record (SQLite EHR)</label>
            <select id="patientSelect" onchange="loadSelectedPatient()"></select>
          </div>
          <div>
            <label>Active Medications (comma-separated)</label>
            <input id="medications" value="atorvastatin, metformin, amlodipine"/>
          </div>
        </div>
        <div class="chk-grid">
          <label><input type="checkbox" id="hist_smoking" checked/> Smoking</label>
          <label><input type="checkbox" id="hist_diabetes" checked/> Diabetes</label>
          <label><input type="checkbox" id="hist_hypertension" checked/> Hypertension</label>
          <label><input type="checkbox" id="hist_family" checked/> Family History CVD</label>
          <label><input type="checkbox" id="hist_mi"/> Prior MI / Stroke</label>
        </div>
        <div id="visitTimeline" class="meta"></div>
      </div>

      <!-- TAB 3: ONLINE FHIR CONNECTOR -->
      <div id="tab-fhir" class="tab-panel">
        <div class="grid2">
          <div><label>HL7 FHIR R4 Server URL</label><input id="fhirUrl" value="https://hapi.fhir.org/baseR4"/></div>
          <div><label>External FHIR Patient ID</label><input id="fhirPatientId" placeholder="e.g. example or 12345"/></div>
        </div>
        <button class="btn btn-sec" style="margin-top:10px;" onclick="pullFhirPatient()">Pull Patient from FHIR R4</button>
        <div id="fhirStatus" class="meta">Connects to public HAPI FHIR R4 or any hospital EHR endpoint when Online Mode is active.</div>
      </div>
    </section>

    <section class="card">
      <h2>Clinical Biomarkers (13 UCI Features — Auto-Filled or Editable)</h2>
      <div class="grid3">
        <div><label>Age</label><input id="age" type="number" value="62"/></div>
        <div><label>Sex (1=M, 0=F)</label><select id="sex"><option value="1" selected>Male (1)</option><option value="0">Female (0)</option></select></div>
        <div><label>Chest Pain (0-3)</label><select id="cp"><option value="0" selected>Typical (0)</option><option value="1">Atypical (1)</option><option value="2">Non-Anginal (2)</option><option value="3">Asymptomatic (3)</option></select></div>
        <div><label>Resting BP</label><input id="trestbps" type="number" value="152"/></div>
        <div><label>Cholesterol</label><input id="chol" type="number" value="286"/></div>
        <div><label>FBS &gt; 120</label><select id="fbs"><option value="0">No (0)</option><option value="1" selected>Yes (1)</option></select></div>
        <div><label>Resting ECG</label><select id="restecg"><option value="0">Normal (0)</option><option value="1" selected>ST-T Abn (1)</option><option value="2">LVH (2)</option></select></div>
        <div><label>Max HR</label><input id="thalach" type="number" value="124"/></div>
        <div><label>Ex. Angina</label><select id="exang"><option value="0">No (0)</option><option value="1" selected>Yes (1)</option></select></div>
        <div><label>ST Depr (mm)</label><input id="oldpeak" type="number" step="0.1" value="2.3"/></div>
        <div><label>ST Slope</label><select id="slope"><option value="0">Down (0)</option><option value="1" selected>Flat (1)</option><option value="2">Up (2)</option></select></div>
        <div><label>Major Vessels</label><input id="ca" type="number" value="2" min="0" max="3"/></div>
        <div><label>Thalassemia</label><select id="thal"><option value="1">Normal (1)</option><option value="2">Fixed (2)</option><option value="3" selected>Reversible (3)</option></select></div>
      </div>
      <button class="btn btn-primary" onclick="runEvaluation(false)">Evaluate Risk, History, SHAP &amp; Online Evidence</button>
      <button class="btn btn-soft" style="width:100%;margin-top:8px;" onclick="runEvaluation(true)">Evaluate &amp; Save Visit to Patient History</button>
    </section>
  </div>

  <div>
    <section class="card">
      <h2>Risk Stratification (ML Ensemble + Longitudinal History) <span id="riskBadge" class="badge" style="background:#64748B;">Ready</span></h2>
      <div class="risk-strip">
        <div class="risk-box">
          <div class="meta" style="margin:0;">BASE ML ENSEMBLE RISK</div>
          <div class="risk-num" id="baseRisk">--%</div>
        </div>
        <div class="risk-box" style="background:#FFF7ED;">
          <div class="meta" style="margin:0;">HISTORY-ADJUSTED RISK</div>
          <div class="risk-num" id="adjRisk" style="color:var(--coral);">--%</div>
          <div class="meta" id="trendText" style="margin:2px 0 0;"></div>
        </div>
      </div>
      <div id="historyFactors"></div>

      <h2 style="margin-top:14px;">Top Local SHAP Feature Drivers</h2>
      <div id="shapBars"></div>

      <h2 style="margin-top:14px;">Actionable What-If Risk Reductions</h2>
      <div id="cfList"></div>
    </section>

    <section class="card">
      <h2>Online Clinical Platform Enrichment (openFDA &amp; PubMed)</h2>
      <div id="fdaList" class="meta">Run evaluation to check active medications via openFDA Drug Label API.</div>
      <h2 style="margin-top:14px;">Live Cardiology Evidence (NCBI PubMed)</h2>
      <div id="pubmedList" class="meta">Run evaluation to retrieve peer-reviewed literature for top SHAP drivers.</div>
      <div class="meta" id="latencyMeta"></div>
    </section>
  </div>
</main>

<script>
const FIELDS = ["age","sex","cp","trestbps","chol","fbs","restecg","thalach","exang","oldpeak","slope","ca","thal"];
let PATIENTS = [];

function switchTab(id, btn) {
  document.querySelectorAll(".tab-panel").forEach(p => p.classList.remove("active"));
  document.querySelectorAll(".tab-btn").forEach(b => b.classList.remove("active"));
  document.getElementById(id).classList.add("active");
  btn.classList.add("active");
}

async function fetchPatients() {
  const res = await fetch("/api/patients");
  PATIENTS = await res.json();
  const sel = document.getElementById("patientSelect");
  sel.innerHTML = PATIENTS.map(p => `<option value="${p.mrn}">${p.mrn} — ${p.full_name}</option>`).join("");
  loadSelectedPatient();
}

function loadSelectedPatient() {
  const mrn = document.getElementById("patientSelect").value;
  const p = PATIENTS.find(x => x.mrn === mrn);
  if (!p) return;
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
}

async function analyzeReportPayload(image_b64, filename) {
  document.getElementById("reportStatus").textContent = "Analyzing report image...";
  const res = await fetch("/api/analyze_report", {
    method:"POST", headers:{"Content-Type":"application/json"},
    body:JSON.stringify({ image_b64, filename, online_mode: document.getElementById("onlineMode").checked })
  });
  const data = await res.json();
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
  document.getElementById("reportStatus").innerHTML = `<strong>${data.engine}:</strong> Extracted <strong>${count}</strong> clinical biomarkers from <em>${data.filename}</em> and auto-filled the form.`;
  runEvaluation(false);
}

function analyzeUploadedReport() {
  const file = document.getElementById("reportFile").files[0];
  if (!file) { alert("Please select a lab report image or PDF first."); return; }
  const reader = new FileReader();
  reader.onload = () => analyzeReportPayload(reader.result, file.name);
  reader.readAsDataURL(file);
}

async function loadSampleReport() {
  const res = await fetch("/api/sample_report");
  const data = await res.json();
  await analyzeReportPayload("data:image/png;base64," + data.image_b64, "sample_lab_report.png");
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
    document.getElementById("fhirStatus").innerHTML = `<strong>Connected:</strong> Loaded <em>${data.full_name}</em> from ${data.server} (${Object.keys(data.mapped_features||{}).length} observations mapped).`;
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
  const payload = {
    mrn: document.getElementById("patientSelect").value || "CP-1042",
    features, history, save_visit: saveVisit,
    online_mode: document.getElementById("onlineMode").checked
  };
  const res = await fetch("/api/predict", { method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify(payload) });
  const data = await res.json();

  document.getElementById("baseRisk").textContent = (data.history_eval.base_ml_risk * 100).toFixed(1) + "%";
  document.getElementById("adjRisk").textContent = (data.history_eval.adjusted_risk * 100).toFixed(1) + "%";
  const td = data.history_eval.trend_vs_last_visit;
  document.getElementById("trendText").textContent = td !== null ? `Trajectory vs last visit: ${td >= 0 ? "+" : ""}${(td*100).toFixed(1)} pts` : "First recorded visit";

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
    const col = s.shap >= 0 ? "#E36414" : "#1B9AAA";
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
  if (saveVisit) fetchPatients();
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
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        if self.path == "/" or self.path.startswith("/?"):
            raw = HTML_UI.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)
        elif self.path == "/health":
            self._send_json({"status": "ok", "service": "CardioPulse XAI v2.0", "threshold": CLIN_THR})
        elif self.path == "/api/patients":
            self._send_json(EHR_DB.list_patients())
        elif self.path == "/api/sample_report":
            b64 = base64.b64encode(SAMPLE_REPORT_PATH.read_bytes()).decode("ascii")
            self._send_json({"filename": "sample_lab_report.png", "image_b64": b64})
        else:
            self._send_json({"error": "Not found"}, status=404)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length).decode("utf-8")) if length > 0 else {}

        if self.path == "/api/analyze_report":
            result = analyze_report_payload(
                image_b64=body.get("image_b64"),
                raw_text=body.get("raw_text"),
                filename=body.get("filename", "report.png"),
                online_mode=bool(body.get("online_mode", False)),
            )
            self._send_json(result)
            return

        if self.path == "/api/patients":
            saved = EHR_DB.upsert_patient(body)
            self._send_json(saved)
            return

        if self.path == "/api/connect/fhir":
            res = fetch_fhir_patient_bundle(
                patient_id=str(body.get("patient_id", "example")),
                fhir_base_url=body.get("server"),
                online_mode=bool(body.get("online_mode", True)),
            )
            self._send_json(res)
            return

        if self.path == "/api/predict":
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
    print(f"CardioPulse XAI v2.0 running on http://{host}:{port}  (Local browser: http://127.0.0.1:{port})")
    server.serve_forever()


if __name__ == "__main__":
    run_server()
