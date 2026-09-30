"""PC probe - Phase 3. Runs every 5s: ping/jitter/loss + DNS, POST to /ingest as probe_id='pc'.
Every 5 min: HTTP download test for down_mbps. Run: .\\.venv\\Scripts\\python.exe backend/pc_probe.py
Backend must be running first.
"""
import subprocess, re, time, socket, sys, os
from collections import deque

try:
    import requests
except ImportError:
    print("need requests: pip install requests")
    sys.exit(1)

SERVER_URL = os.environ.get("SERVER_URL", "http://127.0.0.1:8000/ingest")
PROBE_ID = "pc"
PING_TARGET = "8.8.8.8"
PING_COUNT = 4          # 4 keeps cycle ~5s on Windows (1s per ping)
CYCLE_S = 5
SPEEDTEST_EVERY_S = 300 # 5 min
DOWNLOAD_URL = "https://proof.ovh.net/files/10Mb.dat"  # 10 MB test file
DNS_NAMES = ["google.com","cloudflare.com","wikipedia.org","github.com",
             "microsoft.com","amazon.com","bbc.com","apple.com"]
dns_idx = 0
buf = deque(maxlen=200)
last_speed = 0.0
last_down = None

def measure_ping():
    """Returns (mean_ms, jitter_ms, loss_pct). Windows ping -n."""
    try:
        out = subprocess.run(
            ["ping","-n",str(PING_COUNT),"-w","1000",PING_TARGET],
            capture_output=True, text=True, timeout=15).stdout
        times = [float(x) for x in re.findall(r"time[=<](\d+)ms", out)]
        # packet loss line: Lost = X (Y% loss)
        m = re.search(r"Lost = \d+ \((\d+)% loss\)", out)
        loss = float(m.group(1)) if m else (100.0*(PING_COUNT-len(times))/PING_COUNT if times else 100.0)
        if not times:
            return None, None, 100.0
        mean = sum(times)/len(times)
        jitter = sum(abs(times[i]-times[i-1]) for i in range(1,len(times)))/(len(times)-1) if len(times)>1 else 0.0
        return mean, jitter, loss
    except Exception as e:
        print("ping err:", e)
        return None, None, 100.0

def measure_dns():
    global dns_idx
    name = DNS_NAMES[dns_idx]; dns_idx = (dns_idx+1)%len(DNS_NAMES)
    t0 = time.time()
    try:
        socket.gethostbyname(name)
        return (time.time()-t0)*1000.0
    except Exception:
        return None

def measure_rssi():
    """Best-effort RSSI from netsh (Windows). Returns dBm approx or None."""
    try:
        out = subprocess.run(["netsh","wlan","show","interfaces"],
            capture_output=True, text=True, timeout=10).stdout
        m = re.search(r"Signal\s*:\s*(\d+)%", out)
        if m:
            pct = int(m.group(1))
            return pct/2 - 100  # 80% -> -60 dBm approx
    except Exception:
        pass
    return None

def measure_download():
    try:
        t0 = time.time()
        r = requests.get(DOWNLOAD_URL, stream=True, timeout=30)
        total = 0
        for chunk in r.iter_content(chunk_size=256*1024):
            total += len(chunk)
            if time.time()-t0 > 25 or total > 10*1024*1024:
                break
        dt = time.time()-t0
        if dt > 0 and total > 0:
            return total*8/1e6/dt
    except Exception as e:
        print("download err:", e)
    return None

def post(payload):
    try:
        r = requests.post(SERVER_URL, json=payload, timeout=5)
        return r.status_code == 200
    except Exception:
        return False

print(f"PC probe -> {SERVER_URL} every {CYCLE_S}s. Ctrl+C to stop.")
while True:
    cycle0 = time.time()
    ping, jitter, loss = measure_ping()
    dns = measure_dns()
    rssi = measure_rssi()
    down = None
    if time.time()-last_speed >= SPEEDTEST_EVERY_S:
        print("download test...")
        down = measure_download()
        if down is not None:
            last_down = down
            last_speed = time.time()
            print(f"down {down:.1f} Mbps")
        else:
            # reuse last known so classifier still has value, else None
            down = last_down
    payload = {"probe_id": PROBE_ID, "rssi": rssi, "ping_ms": ping,
               "jitter_ms": jitter, "loss_pct": loss, "dns_ms": dns,
               "down_mbps": down, "up_mbps": None}
    if post(payload):
        while buf:  # flush old
            if not post(buf[0]): break
            buf.popleft()
    else:
        buf.append(payload)
        print("POST failed, buffered", len(buf))
    print(f"ping={ping} jitter={jitter} loss={loss}% dns={dns} rssi={rssi} down={down} buf={len(buf)}")
    time.sleep(max(0.1, CYCLE_S-(time.time()-cycle0)))
