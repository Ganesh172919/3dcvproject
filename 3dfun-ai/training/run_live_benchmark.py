import requests
import json
import time
import os

base = "http://127.0.0.1:8010/api/v1"
s = requests.Session()

print("Step 1: Logging in via Demo Account (demo@3dfun.dev)...")
login_res = s.post(f"{base}/auth/login", json={"email": "demo@3dfun.dev", "password": "DemoPassword123!"})
login_json = login_res.json()
if not login_json.get("success"):
    print("Login error:", login_json)
    exit(1)
token = login_json.get("data", {}).get("access_token")
headers = {"Authorization": f"Bearer {token}"}
print(f"Auth Token obtained: {token[:16]}...")

# Check wallet before
wallet_before = s.get(f"{base}/billing/wallet", headers=headers).json().get("data", {})
print(f"Initial Credit Balance: {wallet_before.get('balance')} credits")

img_path = "web/public/examples/rubber-duck-input.png"
if not os.path.exists(img_path):
    img_path = "3dfun-ai/web/public/examples/rubber-duck-input.png"

print(f"Step 2: Uploading custom benchmark image ({img_path})...")
with open(img_path, "rb") as f:
    up_res = s.post(f"{base}/uploads/", files={"file": ("rubber-duck.png", f, "image/png")}, headers=headers)
upload_data = up_res.json().get("data", {})
upload_id = upload_data.get("id")
print(f"Upload complete: ID={upload_id}, Size={upload_data.get('width')}x{upload_data.get('height')}")

print("Step 3: Triggering 3D reconstruction pipeline (Draft 64^3, Local Depth Anything V2)...")
payload = {
    "upload_id": upload_id,
    "quality": "draft",
    "options": {
        "engine": "local",
        "texture_quality": "512",
        "symmetry": True,
        "turntable": False,
        "formats": ["glb", "obj", "stl"]
    },
    "idempotency_key": f"live-duck-bench-{int(time.time())}"
}
gen_res = s.post(f"{base}/generations", json=payload, headers=headers)
gen_data = gen_res.json().get("data", {})
gen_id = gen_data.get("id")
print(f"Generation Job Registered: ID={gen_id}, Initial Status={gen_data.get('status')}")

print("Step 4: Monitoring real-time reconstruction telemetry...")
t0 = time.time()
while time.time() - t0 < 90:
    poll = s.get(f"{base}/generations/{gen_id}", headers=headers).json().get("data", {})
    status = poll.get("status")
    progress = poll.get("progress")
    stage = poll.get("stage")
    print(f"  [{time.time()-t0:04.1f}s] Progress: {progress:3d}% | Stage: {stage:<15} | Status: {status}")
    if status in ["COMPLETED", "FAILED"]:
        break
    time.sleep(2)

duration = round(time.time() - t0, 2)
asset_id = poll.get("asset_id")

print("=" * 60)
print(f"BENCHMARK COMPLETED: Status={status} in {duration}s")
if asset_id:
    asset = s.get(f"{base}/assets/{asset_id}", headers=headers).json().get("data", {})
    print(f"  Asset ID:          {asset_id}")
    print(f"  Polygons:          {asset.get('polygons'):,}")
    print(f"  Vertices:          {asset.get('vertices'):,}")
    print(f"  Watertight 2-Man:  {asset.get('watertight')}")
    print(f"  Model URL:         {asset.get('model_url')}")
    print(f"  Download formats:  {[d.get('format') for d in asset.get('downloads', [])]}")

wallet_after = s.get(f"{base}/billing/wallet", headers=headers).json().get("data", {})
print(f"Post-Run Credit Balance: {wallet_after.get('balance')} credits (Debited {wallet_before.get('balance') - wallet_after.get('balance')})")
print("=" * 60)
