# CardioPulse XAI v2.0 — Hybrid Edge + Cloud Multimodal Healthcare AI

An advanced, competition-ready cardiovascular risk stratification platform that works **both offline on your Lenovo IdeaPad Slim 5i Ultra (Intel Core Ultra 5 + Intel Arc)** and **online in the cloud** (Render, Railway, Hugging Face Spaces, or Docker), with **medical report image analysis**, **longitudinal patient history**, and **live external clinical platform connectors**.

## What's New in v2.0
1. **Multimodal Report Image & PDF Analyzer (`src/report_ocr.py`):** Upload a photo/scan of a lab report or ECG summary (`.png`, `.jpg`, `.pdf`) to automatically extract the 13 clinical biomarkers, comorbidities, and medications. Uses **Local Pillow + Tesseract/pdfplumber + Clinical Regex** when offline, and upgrades to **Gemini Vision API (`gemini-2.5-flash`)** when online with `GEMINI_API_KEY`. Includes a 1-click Demo Lab Report PNG (`assets/sample_lab_report.png`).
2. **Longitudinal Patient History & EHR Memory (`src/patient_history.py`):** Local SQLite EHR database (`data/cardiopulse_ehr.db`) tracking multi-visit risk trajectories, smoking/diabetes/hypertension/family history, prior MI/stroke, and active medications, with a transparent **Base ML Risk vs. History-Adjusted Risk** breakdown.
3. **Hybrid Online Clinical Connectors (`src/connectors.py`):**
   - **HL7 FHIR R4 (`https://hapi.fhir.org/baseR4` or hospital EHR):** Pulls remote `Patient`, `Observation` (LOINC vitals/labs), and `Condition` records.
   - **openFDA Drug Labeling API (`https://api.fda.gov/drug/label.json`):** Checks patient medications for cardiovascular warnings and indications.
   - **NCBI PubMed E-utilities (`https://eutils.ncbi.nlm.nih.gov/entrez/eutils/`):** Fetches live peer-reviewed cardiology literature tailored to each patient's top SHAP risk drivers.
4. **Runs Locally OR Online in the Cloud:** Includes `Dockerfile`, `render.yaml`, `Procfile`, `/health` check, CORS headers, and `0.0.0.0:$PORT` binding.

## Option A — Run Locally on Lenovo IdeaPad Slim 5i Ultra
```bash
cd cardiopulse_xai_v2
pip install -r requirements.txt
python server.py
# Open http://127.0.0.1:8000 in Chrome or Edge
```

## Option B — Deploy Online (Free Cloud Hosting)
### 1. Render.com (One-Click Blueprint via `render.yaml`)
1. Push `cardiopulse_xai_v2/` to a GitHub repository.
2. In [Render](https://render.com/), click **New → Blueprint**, select your repo, and Render reads `render.yaml` + `Dockerfile` automatically (includes Tesseract OCR pre-installed!).
3. Add optional environment variables (`GEMINI_API_KEY`, `OPENFDA_API_KEY`) in Render's Environment tab.
4. Share your live `https://<your-app>.onrender.com` URL with competition judges!

### 2. Hugging Face Spaces (Docker) or Railway
- **Hugging Face Spaces:** Create a new **Docker** Space, upload the folder contents, and set `app_port: 8000` in the Space README YAML.
- **Docker locally/cloud:** `docker build -t cardiopulse-xai . && docker run -p 8000:8000 cardiopulse-xai`

## Run Automated Verification Tests
```bash
python test_v2_pipeline.py
```

> **Clinical Disclaimer:** Academic research and clinical decision-support prototype only. Not a certified medical device.
