# Setup Walkthrough

Written for a fresh machine, Windows-first (with notes for Mac/Linux where
it differs). Most of the rough edges hit during original development are
already fixed in the code — this is the clean path, not a debugging log
(see `docs/failure-log.md` for the actual bugs found and fixed along the way).

## Prerequisites

- **Python 3.11 or 3.12** (avoid 3.13+; some dependencies lag behind newest
  Python releases). Check: `python3 --version`
- **Docker Desktop**. On Windows, this requires **WSL2** — if Docker Desktop
  shows a "WSL not installed" error on first launch, open PowerShell as
  administrator, run `wsl --install`, restart your machine, then relaunch
  Docker Desktop.
- **Git**

## Setup

```bash
git clone <your-repo-url>
cd nimbus-support-copilot

python3 -m venv venv
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

pip install -r requirements.txt
pip install -e .          # makes `app` importable from any directory —
                           # no PYTHONPATH juggling needed
```

Copy the environment template and fill in your Groq key
(free tier at [console.groq.com](https://console.groq.com)):

```bash
cp .env.example .env
# edit .env, set GROQ_API_KEY
```

## Start infrastructure and load data

```bash
docker-compose up -d
docker ps          # confirm nimbus_postgres and nimbus_redis are both "Up (healthy)"
```

Load the 200 synthetic orders:

```bash
# Mac/Linux:
docker exec -i nimbus_postgres psql -U nimbus -d nimbus_support < data/seed_orders.sql
# Windows PowerShell (no native support for `<` redirection):
Get-Content data/seed_orders.sql | docker exec -i nimbus_postgres psql -U nimbus -d nimbus_support
```

Chunk, embed, and load the 20 policy documents (first run downloads the
embedding model, a few minutes):

```bash
python scripts/ingest_policies.py
```

## Run it

```bash
uvicorn app.main:app --reload
```

In a second terminal, log in and send a test message:

```bash
# Mac/Linux (curl):
curl -X POST http://127.0.0.1:8000/auth/token -H "Content-Type: application/json" -d '{"customer_id": "CUST-0002"}'

# Windows PowerShell:
$response = Invoke-RestMethod -Uri "http://127.0.0.1:8000/auth/token" -Method Post -Body (@{customer_id="CUST-0002"} | ConvertTo-Json) -ContentType "application/json"; $token = $response.access_token; $headers = @{ Authorization = "Bearer $token" }
$body = @{ session_id=[guid]::NewGuid().ToString(); message="What's your return policy?"; turn_index=0 } | ConvertTo-Json
Invoke-RestMethod -Uri "http://127.0.0.1:8000/chat" -Method Post -Body $body -ContentType "application/json" -Headers $headers
```

`CUST-0002` is a real customer in the seed data who owns `ORD-00174`, useful
for testing the order-status path and the authorization check (query an
order that isn't `ORD-00174` while logged in as `CUST-0002` — it should
escalate, not leak another customer's order).

## Run the tests

```bash
pytest tests/              # 21 tests, mocked providers, no API calls, seconds to run
python eval/run_eval.py    # 29 cases, REAL Groq calls, costs money, takes minutes
```

If you only want to re-check specific eval cases instead of paying for all
29 again:

```bash
python eval/run_eval.py --ids rag-04,rag-06
```

## If something breaks

Check `docs/failure-log.md` first — there's a real chance whatever you hit
is already documented there with the exact fix, since most of the rough
edges in getting this running for the first time already happened once
during development.
