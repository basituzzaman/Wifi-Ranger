#include <WiFi.h>
#include <HTTPClient.h>
#include <ArduinoJson.h>
#include <ESP32Ping.h>
#include <time.h>
#include <math.h>

// ====== EDIT THESE ======
const char* WIFI_SSID   = "YOUR_2G_SSID";   // must be a 2.4 GHz network
const char* WIFI_PASS   = "YOUR_WIFI_PASSWORD";
const char* SERVER_URL = "http://192.168.X.X:8000/ingest";  // your PC's IPv4
const char* PROBE_ID    = "esp32-1";
// ========================

const IPAddress PING_TARGET(8, 8, 8, 8);   // IP, so ping does not depend on DNS
const int  PING_COUNT   = 5;               // pings per cycle
const unsigned long CYCLE_MS = 5000;       // one measurement every 5 s

// Rotating domains so lwIP's small DNS cache rarely answers for us
const char* DNS_NAMES[] = {
  "google.com", "cloudflare.com", "wikipedia.org", "github.com",
  "microsoft.com", "amazon.com", "bbc.com", "apple.com"
};
const int DNS_N = sizeof(DNS_NAMES) / sizeof(DNS_NAMES[0]);
int dnsIdx = 0;

struct Reading {
  double ts;      // epoch seconds, 0 if NTP not synced
  float rssi, ping, jitter, loss, dns;   // NaN = no value
};

const int BUF_MAX = 200;
Reading buf[BUF_MAX];
int bufCount = 0;

unsigned long lastCycle = 0;

void connectWiFi() {
  WiFi.mode(WIFI_STA);
  WiFi.setAutoReconnect(true);
  WiFi.begin(WIFI_SSID, WIFI_PASS);
  Serial.print("Connecting to Wi-Fi");
  unsigned long start = millis();
  while (WiFi.status() != WL_CONNECTED && millis() - start < 20000) {
    delay(500);
    Serial.print(".");
  }
  Serial.println();
  if (WiFi.status() == WL_CONNECTED) {
    Serial.print("Connected. ESP32 IP: ");
    Serial.println(WiFi.localIP());
    configTime(0, 0, "pool.ntp.org", "time.google.com");  // UTC epoch
  } else {
    Serial.println("Wi-Fi connect failed, will retry.");
  }
}

double nowEpoch() {
  time_t t = time(nullptr);
  return (t > 1700000000) ? (double)t : 0.0;   // 0 = not synced yet
}

float measureDns() {
  IPAddress ip;
  const char* name = DNS_NAMES[dnsIdx];
  dnsIdx = (dnsIdx + 1) % DNS_N;
  unsigned long t0 = millis();
  bool ok = WiFi.hostByName(name, ip);
  unsigned long dt = millis() - t0;
  return ok ? (float)dt : NAN;
}

Reading measure() {
  Reading r;
  r.ts   = nowEpoch();
  r.rssi = (WiFi.status() == WL_CONNECTED) ? WiFi.RSSI() : NAN;
  r.ping = r.jitter = r.loss = r.dns = NAN;

  if (WiFi.status() != WL_CONNECTED) {
    r.loss = 100.0;
    return r;
  }

  float times[PING_COUNT];
  int ok = 0;
  for (int i = 0; i < PING_COUNT; i++) {
    if (Ping.ping(PING_TARGET, 1)) {
      times[ok++] = Ping.averageTime();
    }
    delay(50);
  }
  r.loss = 100.0f * (PING_COUNT - ok) / PING_COUNT;

  if (ok > 0) {
    float sum = 0;
    for (int i = 0; i < ok; i++) sum += times[i];
    r.ping = sum / ok;
    if (ok > 1) {
      float d = 0;
      for (int i = 1; i < ok; i++) d += fabsf(times[i] - times[i - 1]);
      r.jitter = d / (ok - 1);          // mean absolute difference
    } else {
      r.jitter = 0;
    }
  }

  r.dns = measureDns();
  return r;
}

void setNumber(JsonDocument& doc, const char* key, float v) {
  if (isnan(v)) doc[key] = nullptr;
  else doc[key] = v;
}

bool postReading(const Reading& r) {
  if (WiFi.status() != WL_CONNECTED) return false;

  JsonDocument doc;
  doc["probe_id"] = PROBE_ID;
  if (r.ts > 0) doc["ts"] = r.ts;
  setNumber(doc, "rssi", r.rssi);
  setNumber(doc, "ping_ms", r.ping);
  setNumber(doc, "jitter_ms", r.jitter);
  setNumber(doc, "loss_pct", r.loss);
  setNumber(doc, "dns_ms", r.dns);

  String body;
  serializeJson(doc, body);

  HTTPClient http;
  http.setTimeout(3000);
  http.begin(SERVER_URL);
  http.addHeader("Content-Type", "application/json");
  int code = http.POST(body);
  http.end();
  return code == 200;
}

void bufferReading(const Reading& r) {
  if (bufCount == BUF_MAX) {                 // drop oldest
    memmove(&buf[0], &buf[1], sizeof(Reading) * (BUF_MAX - 1));
    bufCount--;
  }
  buf[bufCount++] = r;
}

void flushBuffer() {
  while (bufCount > 0) {
    if (!postReading(buf[0])) return;        // stop at first failure
    memmove(&buf[0], &buf[1], sizeof(Reading) * (bufCount - 1));
    bufCount--;
  }
}

void setup() {
  Serial.begin(115200);
  delay(500);
  connectWiFi();
}

void loop() {
  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("Wi-Fi lost, reconnecting...");
    WiFi.disconnect();
    WiFi.begin(WIFI_SSID, WIFI_PASS);
    delay(2000);
  }

  if (millis() - lastCycle >= CYCLE_MS) {
    lastCycle = millis();

    Reading r = measure();
    Serial.printf("RSSI=%.0f ping=%.1f jitter=%.1f loss=%.0f%% dns=%.0f | buffered=%d\n",
                  r.rssi, r.ping, r.jitter, r.loss, r.dns, bufCount);

    if (postReading(r)) {
      flushBuffer();
    } else {
      bufferReading(r);
      Serial.println("POST failed, reading buffered.");
    }
  }
}
