"""
Longitudinal Patient History, Demographics & Saved Reports EHR Store (SQLite)
for CardioPulse XAI v2.0.
"""
from __future__ import annotations
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
import numpy as np


class PatientHistoryDB:
    def __init__(self, db_path: str | Path = "data/cardiopulse_ehr.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()
        self._seed_demo_patients()

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self):
        with self._conn() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS patients (
                    mrn TEXT PRIMARY KEY,
                    full_name TEXT NOT NULL,
                    phone TEXT DEFAULT '',
                    blood_group TEXT DEFAULT '',
                    smoking INTEGER DEFAULT 0,
                    diabetes INTEGER DEFAULT 0,
                    hypertension INTEGER DEFAULT 0,
                    family_history INTEGER DEFAULT 0,
                    prior_mi_stroke INTEGER DEFAULT 0,
                    medications_json TEXT DEFAULT '[]',
                    notes TEXT DEFAULT '',
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS visits (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    mrn TEXT NOT NULL,
                    visit_date TEXT NOT NULL,
                    features_json TEXT NOT NULL,
                    base_ml_risk REAL NOT NULL,
                    adjusted_risk REAL NOT NULL,
                    tier TEXT NOT NULL,
                    source TEXT DEFAULT 'manual',
                    FOREIGN KEY (mrn) REFERENCES patients(mrn)
                );
                CREATE TABLE IF NOT EXISTS saved_reports (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    mrn TEXT NOT NULL,
                    patient_name TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    filename TEXT DEFAULT '',
                    report_type TEXT DEFAULT 'Clinical Assessment',
                    engine TEXT DEFAULT 'CardioPulse XAI',
                    base_ml_risk REAL NOT NULL,
                    adjusted_risk REAL NOT NULL,
                    tier TEXT NOT NULL,
                    structured_report_json TEXT NOT NULL,
                    features_json TEXT NOT NULL,
                    shap_json TEXT DEFAULT '[]',
                    image_data TEXT DEFAULT '',
                    FOREIGN KEY (mrn) REFERENCES patients(mrn)
                );
            """)
            # Safe schema migration for existing SQLite databases created by earlier v2.0 runs
            cols = {r["name"] for r in conn.execute("PRAGMA table_info(patients)").fetchall()}
            if "phone" not in cols:
                conn.execute("ALTER TABLE patients ADD COLUMN phone TEXT DEFAULT ''")
            if "blood_group" not in cols:
                conn.execute("ALTER TABLE patients ADD COLUMN blood_group TEXT DEFAULT ''")
            rcols = {r["name"] for r in conn.execute("PRAGMA table_info(saved_reports)").fetchall()}
            if "image_data" not in rcols:
                conn.execute("ALTER TABLE saved_reports ADD COLUMN image_data TEXT DEFAULT ''")

    def _seed_demo_patients(self):
        with self._conn() as conn:
            count = conn.execute("SELECT COUNT(*) FROM patients").fetchone()[0]
            if count > 0:
                return
            now = datetime.now(timezone.utc).isoformat()
            demos = [
                ("CP-1042", "Demo Patient A (High-Risk Follow-Up)", "+91-98000-01042", "B+",
                 1, 1, 1, 1, 0, ["atorvastatin", "metformin", "amlodipine"],
                 "Progressive exertional angina over 6 months; elevated BP & lipids."),
                ("CP-1088", "Demo Patient B (Improving Cardiac Rehab)", "+91-98000-01088", "O+",
                 0, 0, 1, 0, 0, ["lisinopril", "aspirin"],
                 "Completed 12-week supervised cardiac rehab; resting BP improved."),
                ("CP-1105", "Demo Patient C (Preventive Screening)", "+91-98000-01105", "A+",
                 0, 0, 0, 1, 0, [],
                 "Routine annual checkup; paternal history of early coronary disease."),
            ]
            for row in demos:
                mrn, name, phone, bg, smk, diab, htn, fam, pmi, meds, notes = row
                conn.execute(
                    """INSERT INTO patients
                       (mrn, full_name, phone, blood_group, smoking, diabetes, hypertension,
                        family_history, prior_mi_stroke, medications_json, notes, updated_at)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (mrn, name, phone, bg, smk, diab, htn, fam, pmi, json.dumps(meds), notes, now),
                )
            visits = [
                ("CP-1042", "2026-03-14", {"age": 61, "sex": 1, "cp": 1, "trestbps": 140, "chol": 258, "fbs": 1, "restecg": 1, "thalach": 138, "exang": 0, "oldpeak": 1.2, "slope": 1, "ca": 1, "thal": 3}, 0.61, 0.72, "High Risk", "clinic"),
                ("CP-1042", "2026-07-22", {"age": 62, "sex": 1, "cp": 0, "trestbps": 146, "chol": 272, "fbs": 1, "restecg": 1, "thalach": 131, "exang": 1, "oldpeak": 1.8, "slope": 1, "ca": 2, "thal": 3}, 0.74, 0.83, "High Risk", "clinic"),
                ("CP-1088", "2026-04-10", {"age": 53, "sex": 0, "cp": 2, "trestbps": 144, "chol": 245, "fbs": 0, "restecg": 0, "thalach": 148, "exang": 0, "oldpeak": 0.9, "slope": 2, "ca": 0, "thal": 1}, 0.34, 0.38, "Moderate Risk", "clinic"),
                ("CP-1088", "2026-09-18", {"age": 53, "sex": 0, "cp": 2, "trestbps": 128, "chol": 212, "fbs": 0, "restecg": 0, "thalach": 162, "exang": 0, "oldpeak": 0.4, "slope": 2, "ca": 0, "thal": 1}, 0.21, 0.24, "Low Risk", "rehab"),
            ]
            for mrn, vdate, feats, base_r, adj_r, tier, src in visits:
                conn.execute(
                    "INSERT INTO visits (mrn, visit_date, features_json, base_ml_risk, adjusted_risk, tier, source) VALUES (?,?,?,?,?,?,?)",
                    (mrn, vdate, json.dumps(feats), base_r, adj_r, tier, src),
                )

    def list_patients(self) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute("SELECT * FROM patients ORDER BY updated_at DESC, mrn ASC").fetchall()
            out = []
            for r in rows:
                d = dict(r)
                d["medications"] = json.loads(d.pop("medications_json"))
                vrows = conn.execute(
                    "SELECT visit_date, base_ml_risk, adjusted_risk, tier, features_json FROM visits WHERE mrn=? ORDER BY visit_date ASC, id ASC",
                    (d["mrn"],),
                ).fetchall()
                d["visits"] = [
                    {
                        "visit_date": v["visit_date"],
                        "base_ml_risk": v["base_ml_risk"],
                        "adjusted_risk": v["adjusted_risk"],
                        "tier": v["tier"],
                        "features": json.loads(v["features_json"]),
                    }
                    for v in vrows
                ]
                rrows = conn.execute(
                    "SELECT id, created_at, filename, report_type, engine, base_ml_risk, adjusted_risk, tier, (LENGTH(COALESCE(image_data, '')) > 0) AS has_image FROM saved_reports WHERE mrn=? ORDER BY id DESC",
                    (d["mrn"],),
                ).fetchall()
                d["saved_reports"] = [dict(rr) for rr in rrows]
                out.append(d)
            return out

    def get_patient(self, mrn: str) -> dict | None:
        if not mrn:
            return None
        for p in self.list_patients():
            if p["mrn"] == mrn:
                return p
        return None

    def upsert_patient(self, data: dict) -> dict:
        now = datetime.now(timezone.utc).isoformat()
        mrn = str(data.get("mrn") or "").strip()
        if not mrn:
            mrn = f"CP-{datetime.now(timezone.utc).strftime('%H%M%S')}"
        meds = data.get("medications", [])
        if isinstance(meds, str):
            meds = [m.strip().lower() for m in meds.split(",") if m.strip()]
        with self._conn() as conn:
            conn.execute(
                """INSERT INTO patients
                   (mrn, full_name, phone, blood_group, smoking, diabetes, hypertension,
                    family_history, prior_mi_stroke, medications_json, notes, updated_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(mrn) DO UPDATE SET
                     full_name=excluded.full_name, phone=excluded.phone, blood_group=excluded.blood_group,
                     smoking=excluded.smoking, diabetes=excluded.diabetes,
                     hypertension=excluded.hypertension, family_history=excluded.family_history,
                     prior_mi_stroke=excluded.prior_mi_stroke, medications_json=excluded.medications_json,
                     notes=excluded.notes, updated_at=excluded.updated_at""",
                (
                    mrn,
                    str(data.get("full_name") or f"Patient {mrn}").strip(),
                    str(data.get("phone") or "").strip(),
                    str(data.get("blood_group") or "").strip(),
                    int(bool(data.get("smoking", False))),
                    int(bool(data.get("diabetes", False))),
                    int(bool(data.get("hypertension", False))),
                    int(bool(data.get("family_history", False))),
                    int(bool(data.get("prior_mi_stroke", False))),
                    json.dumps(meds),
                    str(data.get("notes") or "").strip(),
                    now,
                ),
            )
        return self.get_patient(mrn)

    def record_visit(self, mrn: str, features: dict, base_risk: float, adj_risk: float, tier: str, source: str = "web") -> dict:
        vdate = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO visits (mrn, visit_date, features_json, base_ml_risk, adjusted_risk, tier, source) VALUES (?,?,?,?,?,?,?)",
                (mrn, vdate, json.dumps(features), round(base_risk, 4), round(adj_risk, 4), tier, source),
            )
        return self.get_patient(mrn)

    def save_report(self, payload: dict) -> dict:
        mrn = str(payload.get("mrn") or "").strip()
        if not mrn:
            raise ValueError("Patient MRN is required to save a report.")
        patient = self.get_patient(mrn)
        if not patient:
            patient = self.upsert_patient({"mrn": mrn, "full_name": payload.get("patient_name", f"Patient {mrn}")})

        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        rep = payload.get("structured_report") or {}
        with self._conn() as conn:
            cur = conn.execute(
                """INSERT INTO saved_reports
                   (mrn, patient_name, created_at, filename, report_type, engine,
                    base_ml_risk, adjusted_risk, tier, structured_report_json, features_json, shap_json, image_data)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    mrn,
                    patient["full_name"],
                    now,
                    str(payload.get("filename") or "clinical_assessment"),
                    str(rep.get("report_type") or "Clinical Risk Assessment"),
                    str(rep.get("engine") or "CardioPulse XAI"),
                    round(float(payload.get("base_ml_risk", 0.0)), 4),
                    round(float(payload.get("adjusted_risk", 0.0)), 4),
                    str(payload.get("tier") or "Evaluated"),
                    json.dumps(rep),
                    json.dumps(payload.get("features") or {}),
                    json.dumps(payload.get("shap") or []),
                    str(payload.get("image_data") or ""),
                ),
            )
            report_id = cur.lastrowid
        # Also append a visit record so longitudinal trajectory stays up to date
        if payload.get("features"):
            self.record_visit(
                mrn,
                payload["features"],
                float(payload.get("base_ml_risk", 0.0)),
                float(payload.get("adjusted_risk", 0.0)),
                str(payload.get("tier") or "Evaluated"),
                source="saved_report",
            )
        return self.get_report(report_id)

    def get_report(self, report_id: int) -> dict | None:
        with self._conn() as conn:
            r = conn.execute("SELECT * FROM saved_reports WHERE id=?", (int(report_id),)).fetchone()
            if not r:
                return None
            d = dict(r)
            d["structured_report"] = json.loads(d.pop("structured_report_json"))
            d["features"] = json.loads(d.pop("features_json"))
            d["shap"] = json.loads(d.pop("shap_json"))
            return d

    def list_reports(self, mrn: str | None = None) -> list[dict]:
        with self._conn() as conn:
            if mrn:
                rows = conn.execute("SELECT * FROM saved_reports WHERE mrn=? ORDER BY id DESC", (mrn,)).fetchall()
            else:
                rows = conn.execute("SELECT * FROM saved_reports ORDER BY id DESC LIMIT 50").fetchall()
            out = []
            for r in rows:
                d = dict(r)
                d["structured_report"] = json.loads(d.pop("structured_report_json"))
                d["features"] = json.loads(d.pop("features_json"))
                d["shap"] = json.loads(d.pop("shap_json"))
                out.append(d)
            return out

    def delete_report(self, report_id: int) -> bool:
        with self._conn() as conn:
            cur = conn.execute("DELETE FROM saved_reports WHERE id=?", (int(report_id),))
            return cur.rowcount > 0


def apply_history_modifier(base_ml_risk: float, history: dict | None, prior_visits: list[dict] | None = None) -> dict:
    """Transparent Bayesian log-odds adjustment incorporating longitudinal patient history."""
    p = float(np.clip(base_ml_risk, 1e-4, 1.0 - 1e-4))
    logit = float(np.log(p / (1.0 - p)))
    factors = []

    if history:
        weights = [
            ("prior_mi_stroke", 0.55, "Prior MI / Stroke history"),
            ("smoking", 0.32, "Active tobacco / smoking"),
            ("diabetes", 0.28, "Documented diabetes mellitus"),
            ("hypertension", 0.20, "Chronic hypertension"),
            ("family_history", 0.22, "Family history of premature CVD"),
        ]
        for key, delta_lo, label in weights:
            if history.get(key):
                logit += delta_lo
                factors.append({"factor": label, "log_odds_shift": f"+{delta_lo:.2f}"})

    trend_delta = None
    if prior_visits:
        last_adj = float(prior_visits[-1]["adjusted_risk"])
        adj_preview = float(1.0 / (1.0 + np.exp(-logit)))
        trend_delta = round(adj_preview - last_adj, 4)
        if trend_delta >= 0.08:
            logit += 0.15
            factors.append({"factor": f"Worsening risk trajectory (+{trend_delta*100:.1f} pts vs prior visit)", "log_odds_shift": "+0.15"})

    adj_risk = float(1.0 / (1.0 + np.exp(-logit)))
    return {
        "base_ml_risk": round(base_ml_risk, 4),
        "adjusted_risk": round(adj_risk, 4),
        "history_delta": round(adj_risk - base_ml_risk, 4),
        "trend_vs_last_visit": trend_delta,
        "applied_factors": factors,
    }
