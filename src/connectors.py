"""
Online Clinical Platform Connectors for CardioPulse XAI v2.0 (stdlib urllib, zero bloat).
Gracefully falls back to local offline mode when disconnected or when online_mode=False.

Supported External Platforms:
  1. HL7 FHIR R4 EHR Server (default: https://hapi.fhir.org/baseR4 or custom FHIR_BASE_URL)
  2. openFDA Drug Labeling API (https://api.fda.gov/drug/label.json)
  3. NCBI PubMed E-utilities API (https://eutils.ncbi.nlm.nih.gov/entrez/eutils/)
"""
from __future__ import annotations
import json
import os
import urllib.parse
import urllib.request
from datetime import datetime


def _http_get_json(url: str, timeout: float = 8.0) -> dict | None:
    req = urllib.request.Request(
        url,
        headers={"Accept": "application/json", "User-Agent": "CardioPulse-XAI/2.0"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception:
        return None


def fetch_fhir_patient_bundle(patient_id: str, fhir_base_url: str | None = None, online_mode: bool = True) -> dict:
    """
    Pull Patient demographics, LOINC vitals/labs (BP, Cholesterol, Glucose, HR),
    and Conditions from an HL7 FHIR R4 server (e.g., https://hapi.fhir.org/baseR4).
    """
    base = (fhir_base_url or os.environ.get("FHIR_BASE_URL") or "https://hapi.fhir.org/baseR4").rstrip("/")
    if not online_mode:
        return {
            "status": "offline",
            "server": base,
            "message": "Online mode is disabled. Enable Online Mode to query external HL7 FHIR R4 EHR servers.",
        }

    pat = _http_get_json(f"{base}/Patient/{urllib.parse.quote(patient_id)}")
    if not pat or pat.get("resourceType") != "Patient":
        return {
            "status": "unreachable_or_not_found",
            "server": base,
            "patient_id": patient_id,
            "message": f"Could not retrieve FHIR Patient/{patient_id} from {base}.",
        }

    # Parse demographics
    name_obj = (pat.get("name") or [{}])[0]
    given = " ".join(name_obj.get("given", []))
    family = name_obj.get("family", "")
    full_name = f"{given} {family}".strip() or f"FHIR Patient {patient_id}"
    gender = pat.get("gender", "").lower()
    sex = 1 if gender == "male" else (0 if gender == "female" else None)

    age = None
    bdate = pat.get("birthDate")
    if bdate and len(bdate) >= 4:
        try:
            age = max(18, min(95, datetime.utcnow().year - int(bdate[:4])))
        except ValueError:
            pass

    # Query recent Observations (LOINC: 8480-6 Systolic BP, 2093-3 Cholesterol, 1558-6 Fasting Glucose, 8867-4 Heart Rate)
    obs_bundle = _http_get_json(f"{base}/Observation?patient={urllib.parse.quote(patient_id)}&_count=25") or {}
    mapped = {}
    if age is not None:
        mapped["age"] = age
    if sex is not None:
        mapped["sex"] = sex

    for entry in obs_bundle.get("entry", []):
        res = entry.get("resource", {})
        codings = res.get("code", {}).get("coding", [])
        codes = {c.get("code") for c in codings}
        val = res.get("valueQuantity", {}).get("value")
        if val is None:
            continue
        if "8480-6" in codes and "trestbps" not in mapped:
            mapped["trestbps"] = int(round(val))
        elif "2093-3" in codes and "chol" not in mapped:
            mapped["chol"] = int(round(val))
        elif "1558-6" in codes and "fbs" not in mapped:
            mapped["fbs"] = 1 if val > 120 else 0
        elif "8867-4" in codes and "thalach" not in mapped:
            mapped["thalach"] = int(round(val))

    # Query Conditions for comorbidities
    cond_bundle = _http_get_json(f"{base}/Condition?patient={urllib.parse.quote(patient_id)}&_count=20") or {}
    conditions = []
    for entry in cond_bundle.get("entry", []):
        txt = entry.get("resource", {}).get("code", {}).get("text")
        if txt:
            conditions.append(txt)

    return {
        "status": "connected",
        "server": base,
        "patient_id": patient_id,
        "full_name": full_name,
        "mapped_features": mapped,
        "conditions": conditions[:8],
    }


def fetch_openfda_drug_safety(medications: list[str], online_mode: bool = True) -> list[dict]:
    """Check patient's active medications against the openFDA Drug Labeling API."""
    if not medications:
        return []
    if not online_mode:
        return [{"drug": m, "status": "offline", "summary": "Enable Online Mode to query openFDA drug labeling."} for m in medications[:4]]

    results = []
    api_key = os.environ.get("OPENFDA_API_KEY", "").strip()
    for med in medications[:4]:
        q = urllib.parse.quote(f'openfda.generic_name:"{med}"')
        url = f"https://api.fda.gov/drug/label.json?search={q}&limit=1"
        if api_key:
            url += f"&api_key={urllib.parse.quote(api_key)}"
        data = _http_get_json(url, timeout=6.0)
        if data and data.get("results"):
            item = data["results"][0]
            warn = (item.get("warnings") or item.get("warnings_and_cautions") or item.get("boxed_warning") or ["No boxed warning returned."])[0]
            ind = (item.get("indications_and_usage") or [""])[0]
            results.append({
                "drug": med,
                "status": "live_openfda",
                "indication": ind[:180] + ("..." if len(ind) > 180 else ""),
                "warning": warn[:220] + ("..." if len(warn) > 220 else ""),
                "source_url": "https://open.fda.gov/apis/drug/label/",
            })
        else:
            results.append({
                "drug": med,
                "status": "offline_fallback",
                "summary": "openFDA lookup unavailable in current network environment; verify dosing locally.",
            })
    return results


def fetch_pubmed_evidence(top_features: list[str], online_mode: bool = True) -> list[dict]:
    """Retrieve live peer-reviewed cardiology literature from NCBI PubMed E-utilities."""
    feature_terms = {
        "oldpeak": "ST depression myocardial ischemia",
        "exang": "exercise induced angina coronary artery disease",
        "ca": "coronary artery stenosis angiography",
        "thal": "myocardial perfusion defect",
        "trestbps": "hypertension cardiovascular risk stratification",
        "chol": "hypercholesterolemia cardiovascular risk",
        "thalach": "chronotropic incompetence heart rate reserve",
        "cp": "angina pectoris diagnosis",
    }
    terms = [feature_terms.get(f, "cardiovascular risk") for f in top_features[:2]]
    query = " AND ".join(f"({t})" for t in terms) if terms else "cardiovascular disease risk prediction"

    if not online_mode:
        return [{
            "status": "offline",
            "title": "Online Evidence Search Paused (Local Edge Mode)",
            "journal": "Switch to Hybrid Online Mode to fetch live PubMed citations via NCBI E-utilities.",
            "url": "https://www.nlm.nih.gov/dataguide/eutilities/utilities.html",
        }]

    search_url = (
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?"
        + urllib.parse.urlencode({"db": "pubmed", "term": query, "retmax": 3, "retmode": "json", "sort": "relevance"})
    )
    sdata = _http_get_json(search_url, timeout=6.0)
    ids = (sdata or {}).get("esearchresult", {}).get("idlist", [])
    if not ids:
        return [{
            "status": "offline_fallback",
            "title": "PubMed E-utilities unreachable in current network",
            "journal": "Query ready: " + query,
            "url": "https://pubmed.ncbi.nlm.nih.gov/",
        }]

    sum_url = (
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?"
        + urllib.parse.urlencode({"db": "pubmed", "id": ",".join(ids), "retmode": "json"})
    )
    sumdata = _http_get_json(sum_url, timeout=6.0) or {}
    res_map = sumdata.get("result", {})
    papers = []
    for pmid in ids:
        rec = res_map.get(pmid, {})
        papers.append({
            "status": "live_pubmed",
            "pmid": pmid,
            "title": rec.get("title", "PubMed Article"),
            "journal": f"{rec.get('fulljournalname', 'Journal')} ({rec.get('pubdate', '')})",
            "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
        })
    return papers
