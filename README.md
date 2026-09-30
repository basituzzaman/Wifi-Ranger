# AI Wi-Fi Health Monitor
ESP32 + PC probes → FastAPI + SQLite → IsolationForest + RandomForest → live dashboard.

## Quick run (any PC)
1. `python -m venv .venv` + activate. Windows: `.\.venv\Scripts\Activate.ps1`, Linux/Mac: `source .venv/bin/activate`
2. `pip install -r requirements.txt`
3. `python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000` (allow firewall 8000, fix PC IP e.g. 192.168.0.105)
4. New terminal: `python backend/pc_probe.py` (set `SERVER_URL` env or edit file to `http://<PC-IP>:8000/ingest`)
5. Flash `firmware/probe/probe.ino`: set `WIFI_SSID`, `WIFI_PASS`, `SERVER_URL=http://<PC-IP>:8000/ingest` (2.4 GHz only). Power via USB/powerbank.
6. Verify `http://<PC-IP>:8000/latest` + `/analysis/latest`.
7. Dashboard: serve `dashboard/` via backend static or `python -m http.server` in `dashboard/`, open `http://<PC-IP>:8000` or file. API auto-uses page host.
8. Label faults: `POST /label {"label":"WIFI_WEAK"}` etc. Log `data/notes.txt` start,end,fault.
9. Train: `python ml/prototype.py` → `ml/*.joblib`, `roc_iso.png`. Restart backend to load models live.

## Classes
NORMAL, WIFI_WEAK (far), CONGESTION (2x4K+download), PACKET_LOSS (Clumsy 10% / tc netem), DNS_PROBLEM (dead 192.0.2.1). 300+ rows/class, 3000+ NORMAL.

## Notes
- Time split, no shuffle. Per-label split for classifier (sequential faults).
- Known drift: night→day ping shift; PC-only = WEAK blind; ESP32 2.4 GHz only; artificial faults.
- Do NOT commit `data/*.db`, `.venv/`, `ml/*.joblib` (gitignored). Share `notes.txt` + `dataset_clean.csv` sample only.
