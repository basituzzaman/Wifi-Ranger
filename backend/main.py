import sqlite3, time, json
from pathlib import Path
from typing import Optional
from fastapi import FastAPI
from pydantic import BaseModel

DB = Path(__file__).resolve().parent.parent / "data" / "monitor.db"
DB.parent.mkdir(exist_ok=True)
ML = Path(__file__).resolve().parent.parent / "ml"

app = FastAPI(title="WiFi Monitor")
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
current_label = {"value": "NORMAL"}

# --- live models (loaded if trained) ---
scaler = iso = rf = le = None
THR = 0.62  # 95th pct from prototype val
NORM_MED = {"rssi": -58.0, "ping_ms": 42.0, "jitter_ms": 30.0, "loss_pct": 0.5, "dns_ms": 50.0}
FEAT_ORDER = None
try:
    import joblib, numpy as np
    if (ML/"scaler.joblib").exists():
        scaler = joblib.load(ML/"scaler.joblib")
        iso = joblib.load(ML/"isoforest.joblib")
        rf = joblib.load(ML/"rf.joblib")
        le = joblib.load(ML/"labels.joblib")
        # feature order from scaler
        try:
            FEAT_ORDER = list(scaler.feature_names_in_)
        except Exception:
            FEAT_ORDER = None
        print("live models loaded")
except Exception as e:
    print("live models not loaded:", e)


def db():
    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row
    return conn


with db() as c:
    c.execute("""
    CREATE TABLE IF NOT EXISTS measurements (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ts REAL, probe_id TEXT,
        rssi REAL, ping_ms REAL, jitter_ms REAL, loss_pct REAL,
        dns_ms REAL, down_mbps REAL, up_mbps REAL,
        label TEXT
    )""")
    c.execute("""
    CREATE TABLE IF NOT EXISTS analysis (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ts REAL, probe_id TEXT, score REAL, status TEXT,
        cause TEXT, confidence REAL, evidence TEXT
    )""")


def compute_features(rows):
    """rows: list of dicts oldest->newest (>=3). Returns dict of 41 feats matching prototype."""
    import pandas as pd, numpy as np
    g = pd.DataFrame(rows)
    for col in ["rssi","ping_ms","jitter_ms","loss_pct","dns_ms"]:
        g[col] = pd.to_numeric(g[col], errors="coerce")
    g["dns_ms"] = g["dns_ms"].fillna(5000)
    g[["rssi","ping_ms","jitter_ms","loss_pct"]] = g[["rssi","ping_ms","jitter_ms","loss_pct"]].ffill().bfill()
    out = {}
    for w, tag in [(6,"30s"),(12,"60s")]:
        for col in ["rssi","ping_ms","jitter_ms","loss_pct","dns_ms"]:
            s = g[col].tail(w)
            out[f"{col}_{tag}_mean"] = float(s.mean())
            out[f"{col}_{tag}_std"] = float(s.std() or 0)
            out[f"{col}_{tag}_min"] = float(s.min())
            out[f"{col}_{tag}_max"] = float(s.max())
    x = g["rssi"].tail(6).values
    out["rssi_slope"] = float(np.polyfit(range(len(x)), x, 1)[0]) if len(x) >= 3 else 0.0
    return out


def analyze_probe(probe_id, ts):
    """Run live inference, store + return result. Never raises."""
    try:
        if scaler is None or iso is None:
            return {"ts": ts, "probe_id": probe_id, "score": 0.0, "status": "Normal", "cause": "NORMAL", "confidence": 1.0, "evidence": []}
        with db() as c:
            rows = c.execute("SELECT rssi,ping_ms,jitter_ms,loss_pct,dns_ms FROM measurements WHERE probe_id=? ORDER BY ts DESC LIMIT 12", (probe_id,)).fetchall()
        if len(rows) < 3:
            return {"ts": ts, "probe_id": probe_id, "score": 0.0, "status": "Normal", "cause": "NORMAL", "confidence": 1.0, "evidence": []}
        rows = [dict(r) for r in reversed(rows)]
        feats = compute_features(rows)
        import numpy as np
        order = FEAT_ORDER or sorted(feats.keys())
        import pandas as pd
        Xdf = pd.DataFrame([[feats[k] for k in order]], columns=order)
        Xs = scaler.transform(Xdf)
        score = float(-iso.score_samples(Xs)[0])
        cur = rows[-1]
        ev = []
        for k, label_txt in [("loss_pct", "Packet loss"), ("ping_ms", "Ping"), ("jitter_ms", "Jitter"), ("dns_ms", "DNS"), ("rssi", "RSSI")]:
            v = cur.get(k)
            try: v = float(v)
            except: continue
            base = NORM_MED.get(k)
            if k == "loss_pct" and v > base + 5: ev.append(f"{label_txt} {v:.0f}% vs normal {base:.1f}%")
            elif k == "ping_ms" and v > base + 30: ev.append(f"{label_txt} {v:.0f}ms vs normal {base:.0f}ms")
            elif k == "dns_ms" and (v is None or v > 500 or (v or 0) > base + 100): ev.append(f"{label_txt} {v if v is not None else 'timeout'} vs normal {base:.0f}ms")
            elif k == "rssi" and v < -75: ev.append(f"{label_txt} {v:.0f}dBm vs normal {base:.0f}dBm (weak)")
        if score < THR:
            status, cause, conf = "Normal", "NORMAL", 0.95
        else:
            probs = rf.predict_proba(Xs)[0]
            idx = int(probs.argmax()); conf = float(probs[idx])
            cause = str(le.inverse_transform([idx])[0]) if conf >= 0.5 else "Unknown"
            status = "Degraded" if score < THR * 1.5 else "Severe"
        res = {"ts": ts, "probe_id": probe_id, "score": round(score, 3), "status": status, "cause": cause, "confidence": round(conf, 3), "evidence": ev[:4]}
        with db() as c:
            c.execute("INSERT INTO analysis (ts,probe_id,score,status,cause,confidence,evidence) VALUES (?,?,?,?,?,?,?)",
                      (ts, probe_id, res["score"], status, cause, res["confidence"], json.dumps(ev[:4])))
        return res
    except Exception as e:
        print("analyze err:", e)
        return {"ts": ts, "probe_id": probe_id, "score": 0.0, "status": "Normal", "cause": "NORMAL", "confidence": 0.0, "evidence": []}


class Measurement(BaseModel):
    probe_id: str
    ts: Optional[float] = None
    rssi: Optional[float] = None
    ping_ms: Optional[float] = None
    jitter_ms: Optional[float] = None
    loss_pct: Optional[float] = None
    dns_ms: Optional[float] = None
    down_mbps: Optional[float] = None
    up_mbps: Optional[float] = None


class Label(BaseModel):
    label: str


@app.post("/ingest")
def ingest(m: Measurement):
    ts = m.ts or time.time()
    with db() as c:
        c.execute(
            """INSERT INTO measurements
            (ts, probe_id, rssi, ping_ms, jitter_ms, loss_pct, dns_ms, down_mbps, up_mbps, label)
            VALUES (?,?,?,?,?,?,?,?,?,?)""",
            (ts, m.probe_id, m.rssi, m.ping_ms, m.jitter_ms, m.loss_pct,
             m.dns_ms, m.down_mbps, m.up_mbps, current_label["value"]),
        )
    res = analyze_probe(m.probe_id, ts)
    return {"ok": True, "label": current_label["value"], "analysis": res}


@app.get("/latest")
def latest(probe_id: Optional[str] = None):
    q = "SELECT * FROM measurements"
    args = []
    if probe_id:
        q += " WHERE probe_id=?"
        args.append(probe_id)
    q += " ORDER BY ts DESC LIMIT 1"
    with db() as c:
        row = c.execute(q, args).fetchone()
    return dict(row) if row else {}


@app.get("/history")
def history(hours: float = 1):
    since = time.time() - hours * 3600
    with db() as c:
        rows = c.execute(
            "SELECT * FROM measurements WHERE ts>=? ORDER BY ts", (since,)
        ).fetchall()
    return [dict(r) for r in rows]


@app.get("/anomalies")
def anomalies(hours: float = 24):
    since = time.time() - hours * 3600
    with db() as c:
        rows = c.execute(
            "SELECT * FROM analysis WHERE ts>=? AND status!='Normal' ORDER BY ts DESC LIMIT 200", (since,)
        ).fetchall()
    return [dict(r) for r in rows]


@app.get("/analysis/latest")
def analysis_latest(probe_id: Optional[str] = None):
    q = "SELECT * FROM analysis"
    args = []
    if probe_id:
        q += " WHERE probe_id=?"
        args.append(probe_id)
    q += " ORDER BY ts DESC LIMIT 1"
    with db() as c:
        row = c.execute(q, args).fetchone()
    return dict(row) if row else {}


@app.post("/label")
def set_label(l: Label):
    current_label["value"] = l.label
    return {"current_label": l.label}


@app.get("/label")
def get_label():
    return {"current_label": current_label["value"]}