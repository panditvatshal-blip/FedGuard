# FedGuard — Full-stack federated-learning prototype

FedGuard is a Flask + SQLite website with a real PyTorch training path, local multi-client FedAvg, per-example clipped/noised gradients, persistent model checkpoints, and a verifiable hash-chained audit ledger. It also includes standalone Flower server/client scripts for a networked federation demo.

> **Safety and scope:** Built-in training data are synthetic only. Do not upload or use patient data. This is an educational prototype, not a clinical product or security-certified deployment. The local dashboard performs actual model training on synthetic tabular data. The reported epsilon is a conservative Gaussian-mechanism RDP composition bound without subsampling amplification; it is not a complete privacy assessment for a real dataset or system. The hash-chained ledger is tamper-evident, not a decentralized blockchain.

## Features

- Responsive website, animated 3D background, chart, login/register, and dashboard.
- Flask server-side sessions; passwords are hashed with Werkzeug.
- SQLite persistence for users, training rounds, global model state, DP settings, and audit hashes.
- A small PyTorch binary classifier trained on three isolated synthetic client datasets.
- Per-example gradient clipping followed by Gaussian noise; weighted FedAvg combines client model weights.
- Model checkpoints persist between dashboard rounds so the global model can continue training.
- Privacy accounting uses Gaussian RDP composition with `delta=1e-5` and no sampling-amplification claim.
- `/api/ledger/verify` recalculates block hashes and previous-hash links for the signed-in account.
- Optional Flower network server and client processes using the same PyTorch model and DP local training function.

## Requirements

- Python 3.10 or newer (Python 3.11 is recommended)
- pip
- Internet access in the browser for Google Fonts and Three.js CDN assets
- Installing PyTorch and Flower can take a few minutes and may require a compatible Python/platform build

## Run the web app on Windows (PowerShell)

```powershell
cd path\to\fedguard_fullstack
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
$env:FEDGUARD_SECRET_KEY = "replace-this-with-a-long-random-secret"
python app.py
```

Open `http://127.0.0.1:5000`. On macOS/Linux, activate with `source .venv/bin/activate`.

The SQLite database (`fedguard.db`) is created automatically. Older demo databases are upgraded in place. Keep the database private because it contains account records, password hashes, model checkpoints, and audit metadata.

## Start a training round

1. Sign up and log in.
2. Open the dashboard.
3. Choose the noise multiplier (`σ`) and click **Start round**.
4. The server trains each synthetic client locally with clipped/noised gradients, averages the local weights using FedAvg, evaluates on a separate synthetic set, persists the global checkpoint, and appends a hash-chained audit record.
5. Click **Verify integrity** to recalculate the stored ledger chain.

The `accuracy` and `loss` values are real outputs from the toy model, but they are measured on generated synthetic data and say nothing about clinical performance.

## Optional networked Flower demo

Install the dependencies from `requirements.txt` first. Open separate terminals in the project folder.

Terminal 1 — Flower server:

```powershell
$env:FLOWER_MIN_CLIENTS = "2"
$env:FLOWER_ROUNDS = "3"
$env:DP_NOISE = "1.0"
python flower_server.py
```

Terminal 2 and 3 — connect two clients:

```powershell
python flower_client.py --client-id 0 --server 127.0.0.1:8080
python flower_client.py --client-id 1 --server 127.0.0.1:8080
```

For separate machines, set `--server` to the reachable Flower server host and port, configure firewall rules intentionally, and do not expose this demo server to the public internet. The standalone Flower run is separate from the website's local training endpoint; it is provided for experimenting with real Flower client/server communication. It currently uses synthetic data and does not automatically write its rounds into the website's SQLite ledger.

## API overview

All endpoints are same-origin. POST requests accept JSON.

| Method | Endpoint | Purpose |
| --- | --- | --- |
| `GET` | `/api/health` | Health check and training mode |
| `GET` | `/api/me` | Current session and dashboard |
| `POST` | `/api/register` | Create account and sign in |
| `POST` | `/api/login` | Sign in |
| `POST` | `/api/logout` | Sign out |
| `GET` | `/api/dashboard` | Authenticated user's rounds and ledger |
| `POST` | `/api/rounds` | Run real PyTorch DP-FedAvg round; JSON: `{"noise": 1.0}` |
| `GET` | `/api/ledger/verify` | Verify the authenticated user's hash chain |

## Project structure

```text
fedguard_fullstack/
├── app.py                # Flask API, auth, persistence, ledger verification
├── fedguard_ml.py        # PyTorch model, synthetic clients, DP-SGD, FedAvg
├── flower_server.py      # Optional Flower server
├── flower_client.py      # Optional Flower client node
├── requirements.txt
├── README.md
├── templates/
│   └── index.html
└── fedguard.db           # Created automatically on first run
```

## Production hardening still required

- Strong secret key, HTTPS, secure cookies, CSRF protection, rate limiting, account recovery, monitoring, backups, and a production WSGI server.
- A reviewed privacy accountant and threat model for the exact sampling, participation, clipping, and aggregation scheme.
- Secure aggregation, authenticated/encrypted Flower transport, authorization, node identity, and deployment hardening.
- A production-grade append-only ledger or external anchoring if independent tamper resistance is required.
- Clinical validation and formal privacy/security review before any healthcare use.
