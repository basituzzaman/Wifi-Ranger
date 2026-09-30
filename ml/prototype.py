"""Phase 5-7 prototype: CSV -> clean -> rolling 30s/60s -> time split -> scaler -> IF + RF.
Run: .\\.venv\\Scripts\\python.exe ml\\prototype.py
"""
from pathlib import Path
import sqlite3, joblib
import pandas as pd, numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score,
                             precision_recall_fscore_support)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent.parent
DB = BASE/"data"/"monitor.db"
OUT = BASE/"ml"; OUT.mkdir(exist_ok=True)

con = sqlite3.connect(DB)
df = pd.read_sql("SELECT ts, probe_id, rssi, ping_ms, jitter_ms, loss_pct, dns_ms, down_mbps, up_mbps, label FROM measurements ORDER BY ts", con)
print("rows", len(df), df["label"].value_counts().to_dict())

# clean: drop test probe, clip impossible, fill
df = df[df.probe_id.isin(["esp32-1","pc"])].copy()
for c in ["rssi","ping_ms","jitter_ms","loss_pct","dns_ms","down_mbps"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")
df["loss_pct"] = df["loss_pct"].clip(0,100)
df["rssi"] = df["rssi"].clip(-100,0)
df[["ping_ms","jitter_ms","dns_ms"]] = df[["ping_ms","jitter_ms","dns_ms"]].clip(lower=0)
# dns None (timeout) -> fill 5000 (fault signal) before rolling
df["dns_ms"] = df["dns_ms"].fillna(5000)
df[["rssi","ping_ms","jitter_ms","loss_pct"]] = df[["rssi","ping_ms","jitter_ms","loss_pct"]].ffill().fillna(df.median(numeric_only=True))
df["down_mbps"] = df["down_mbps"].fillna(-1)  # -1 = not measured
df = df.drop_duplicates(subset=["ts","probe_id"]).sort_values("ts").reset_index(drop=True)
df.to_csv(OUT/"dataset_clean.csv", index=False)

# rolling features per probe: 6 rows=30s, 12 rows=60s at 5s
FEATS = ["rssi","ping_ms","jitter_ms","loss_pct","dns_ms"]
rows = []
for pid, g in df.groupby("probe_id"):
    g = g.sort_values("ts").copy()
    for w, tag in [(6,"30s"),(12,"60s")]:
        for c in FEATS:
            r = g[c].rolling(w, min_periods=3)
            g[f"{c}_{tag}_mean"] = r.mean(); g[f"{c}_{tag}_std"] = r.std().fillna(0)
            g[f"{c}_{tag}_min"] = r.min(); g[f"{c}_{tag}_max"] = r.max()
    g["rssi_slope"] = g["rssi"].rolling(6, min_periods=3).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x)>=3 else 0, raw=False).fillna(0)
    rows.append(g)
fall = pd.concat(rows).sort_values("ts")
feat_cols = [c for c in fall.columns if any(k in c for k in ("_mean","_std","_min","_max","slope"))]
f = fall.dropna(subset=feat_cols).reset_index(drop=True)
print("features", len(feat_cols))

# time split 70/15/15 (no shuffle)
n = len(f); i1, i2 = int(n*.7), int(n*.85)
tr, va, te = f.iloc[:i1], f.iloc[i1:i2], f.iloc[i2:]
print(f"split {len(tr)}/{len(va)}/{len(te)}")
sc = StandardScaler().fit(tr[feat_cols])
joblib.dump(sc, OUT/"scaler.joblib")
Xtr, Xva, Xte = sc.transform(tr[feat_cols]), sc.transform(va[feat_cols]), sc.transform(te[feat_cols])

# anomaly: train on NORMAL only
ntr = tr[tr.label=="NORMAL"]
iso = IsolationForest(n_estimators=200, contamination=0.05, random_state=0).fit(sc.transform(ntr[feat_cols]))
joblib.dump(iso, OUT/"isoforest.joblib")
for name, X, d in [("val",Xva,va),("test",Xte,te)]:
    s = -iso.score_samples(X)  # higher = more anomalous
    y = (d.label!="NORMAL").astype(int)
    thr = np.percentile(-iso.score_samples(Xva[va.label=="NORMAL"]), 95)
    pred = (s>=thr).astype(int)
    p,r,f1,_ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)
    try: auc = roc_auc_score(y, s)
    except: auc = float("nan")
    print(f"IF {name}: thr={thr:.3f} P={p:.3f} R={r:.3f} F1={f1:.3f} AUC={auc:.3f}")
    if name=="test":
        # ROC plot
        from sklearn.metrics import roc_curve
        fpr,tpr,_ = roc_curve(y, s)
        plt.figure(); plt.plot(fpr,tpr); plt.plot([0,1],[0,1],"--")
        plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title(f"IF ROC AUC={auc:.3f}")
        plt.savefig(OUT/"roc_iso.png"); plt.close()

# classifier 5-class: per-label time split (70/15/15 within each label) so train sees all faults
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder().fit(f.label); joblib.dump(le, OUT/"labels.joblib")
trs, vas, tes = [], [], []
for lab, g in f.sort_values("ts").groupby("label"):
    g = g.sort_values("ts"); n = len(g)
    a, b = int(n*.7), int(n*.85)
    trs.append(g.iloc[:a]); vas.append(g.iloc[a:b]); tes.append(g.iloc[b:])
ctr, cva, cte = pd.concat(trs), pd.concat(vas), pd.concat(tes)
print(f"clf split {len(ctr)}/{len(cva)}/{len(cte)}", cte.label.value_counts().to_dict())
Xctr, Xcva, Xcte = sc.transform(ctr[feat_cols]), sc.transform(cva[feat_cols]), sc.transform(cte[feat_cols])
ytr, yva, yte = le.transform(ctr.label), le.transform(cva.label), le.transform(cte.label)
rf = RandomForestClassifier(n_estimators=300, min_samples_leaf=2, class_weight="balanced", random_state=0, n_jobs=-1)
rf.fit(Xctr, ytr); joblib.dump(rf, OUT/"rf.joblib")
for name, X, y in [("val",Xcva,yva),("test",Xcte,yte)]:
    p = rf.predict(X)
    print(f"RF {name}:\n", classification_report(y, p, labels=list(range(len(le.classes_))), target_names=list(le.classes_), zero_division=0))
    if name=="test":
        print("confusion:\n", confusion_matrix(y, p, labels=list(range(len(le.classes_)))))
        imp = pd.Series(rf.feature_importances_, index=feat_cols).sort_values(ascending=False)[:10]
        print("top features:\n", imp)
print("saved to ml/")
