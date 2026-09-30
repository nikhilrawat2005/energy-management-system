"""Quick HTTP API test for Ollama - after model is warmed up."""
import json, urllib.request, time

url = "http://localhost:11434/api/generate"
payload = json.dumps({
    "model": "qwen2.5:1.5b",
    "prompt": "Microgrid: Solar=4.2kW, Load=3.1kW, SoC=68%, Tariff=PEAK. Action taken: CHARGE_SOLAR. Give 1-sentence analysis then RECOMMEND:",
    "stream": False,
    "options": {"temperature": 0.2, "num_predict": 60, "top_k": 20}
}).encode()

print("Sending request to Ollama HTTP API...")
t0 = time.time()
req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"}, method="POST")
with urllib.request.urlopen(req, timeout=120) as resp:
    result = json.loads(resp.read().decode())
latency = (time.time() - t0) * 1000
print(f"Latency: {latency:.0f}ms")
print(f"Response: {result.get('response', '???')}")
