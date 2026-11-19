# CardioPulse XAI — Hybrid Edge + Cloud Multimodal Healthcare AI

An explainable cardiovascular risk stratification platform built from scratch in Python that runs **both offline on standard CPU laptops/workstations** (no dedicated GPU required) and **online in the cloud** (Render, Railway, Hugging Face Spaces, or Docker).

## Key Capabilities
1. **ECG Waveform & Lab Report Analyzer (`src/report_ocr.py`):** Upload an ECG strip image, lab report photo, or PDF (`.png`, `.jpg`, `.pdf`). Automatically digitizes ECG waveforms offline (estimating heart rate, R-R rhythm regularity, and ST-segment morphology), parses printed lab text via Pillow + Tesseract/pdfplumber, and generates a detailed structured diagnostic report. Optionally connects to Gemini Vision (`gemini-2.5-flash`) when `GEMINI_API_KEY` is configured.
2. **Longitudinal Patient History & EHR Memory (`src/patient_history.py`):** Local SQLite EHR database (`data/cardiopulse_ehr.db`) tracking multi-visit risk trajectories, comorbidities, and active medications with a transparent **Base ML Risk vs. History-Adjusted Risk** breakdown.
3. **Hybrid Online Clinical Connectors (`src/connectors.py`):**
   - **HL7 FHIR R4 (`https://hapi.fhir.org/baseR4` or custom EHR):** Pulls remote `Patient`, `Observation`, and `Condition` records.
   - **openFDA Drug Labeling API (`https://api.fda.gov/drug/label.json`):** Checks active medications for cardiovascular warnings.
   - **NCBI PubMed E-utilities (`https://eutils.ncbi.nlm.nih.gov/entrez/eutils/`):** Retrieves peer-reviewed cardiology literature tailored to top SHAP risk drivers.
4. **Cloud-Ready Deployment:** Includes `Dockerfile`, `render.yaml`, `Procfile`, `/health` endpoint, CORS support, and `0.0.0.0:$PORT` binding.

## Option A — Run Locally (CPU-Only, Offline Ready)
```bash
pip install -r requirements.txt
python server.py
# Open http://127.0.0.1:8000 in your browser
```

## Option B — Deploy Online (Cloud Hosting)
### Render.com (Blueprint via `render.yaml`)
1. Push this repository to GitHub.
2. In [Render](https://render.com/), choose **New → Blueprint**, connect your repository, and deploy using `render.yaml` + `Dockerfile` (includes Tesseract OCR pre-installed).
3. Optionally add `GEMINI_API_KEY` or `OPENFDA_API_KEY` in the Environment settings.

### Docker / Hugging Face Spaces
```bash
docker build -t cardiopulse-xai .
docker run -p 8000:8000 cardiopulse-xai
```

## Run Automated Verification Tests
```bash
python test_v2_pipeline.py
```

> **Clinical Disclaimer:** Academic research and clinical decision-support prototype only. Not a certified medical device.
