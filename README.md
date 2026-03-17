# 🧠 MindGuard: Commercial Platform Edition (v3)

> *A privacy-preserving, multi-tenant emotional wellness intelligence platform for organizations, schools, and workplaces.*

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: Commercial](https://img.shields.io/badge/License-Commercial-yellow.svg)](LICENSE)
[![Ethics First](https://img.shields.io/badge/Ethics-First-green.svg)](#privacy--ethics-guarantee)

MindGuard has been upgraded from a single-school research tool into a **fully scalable SaaS architecture** ready for commercial licensing and deployment. It provides anonymized aggregate trend analysis across 8 emotional dimensions while strictly protecting individual privacy.

---

## 🌟 New Commercial Upgrades (v3)

1. **Multi-Tenant Architecture**: A robust SQLite layer securely isolates organization data. Manage infinite clients from a single deployment.
2. **API Key Authentication**: Secure access management. Organizations can only view and submit data using their unique API tokens.
3. **Admin Control Panel**: A dedicated internal Gradio dashboard (`app/admin_dashboard.py`) for generating client credentials, tracking platform-wide billing, and managing subscription tiers.
4. **Subscription Tier limits**: Rate limit enforcement on the FastAPI backend based on an organization's pricing plan (Free, Starter, Professional, Enterprise).
5. **Historical Trend Snapshots**: Automated logging of community health scores and critical flags over time, viewable directly from the organization's dashboard.
6. **Docker Ready**: Deploy your entire platform (REST API, Client Dashboard, Admin Dashboard) seamlessly using the new `Dockerfile`.

---

## 🏗️ Platform Stack

```
mindguard/
├── db/                        # NEW: Database layer
│   └── database.py            # SQLite multi-org API, billing tracking, historical snapshots
├── app/                       
│   ├── admin_dashboard.py     # NEW: Internal UI for managing clients & API Keys
│   └── community_dashboard.py # UPGRADED: Client-facing UI (requires API key auth)
├── api/
│   └── community_api.py       # UPGRADED: Multi-tenant REST API with rate limiting
├── config/
│   └── community_config.yaml  # UPGRADED: Pricing tiers, webhooks, 8 emotional dimensions
├── model/                     # Core Mental-BERT and heuristic engines 
└── Dockerfile                 # NEW: One-click cloud deployment pipeline
```

---

## 🚀 Getting Started (Commercial Deployment)

### 1. Local Run
Install the new requirements, then boot the admin stack first:
```bash
# Install core
pip install -r requirements.txt
# Install new commercial web dependencies
pip install fastapi uvicorn httpx plotly gradio pandas

# 1. Start the Admin Control Panel (port 7861)
python app/admin_dashboard.py
```
*Open `http://localhost:7861`. Use this UI to generate your first Organization and API Key.*

```bash
# 2. Start the Organization REST API (port 8000)
uvicorn api.community_api:app --host 0.0.0.0 --port 8000

# 3. Start the Client Dashboard (port 7862)
python app/community_dashboard.py
```
*Open `http://localhost:7862`. Paste the API key you generated in Step 1 to log in and view trends.*

### 2. Docker Cloud Deployment
Deploy all 3 pieces of infrastructure instantly:
```bash
docker build -t mindguard-platform .
docker run -p 8000:8000 -p 7861:7861 -p 7862:7862 mindguard-platform
```

---

## 💰 Subscription Features (`community_config.yaml`)

MindGuard automatically restricts feature access and usage limits based on your defined pricing tiers. 

- **Community Free**: Basic batching and real-time inference.
- **Starter**: Unlocks `Trend History` tracking and larger batch sizes.
- **Professional**: Enables Webhooks, email alerts, and massive API limits.
- **Enterprise**: Fully unlimited integration usage.

---

## 🔒 Privacy & Ethics Guarantee (Selling Point)

Your strongest commercial asset is MindGuard's **Zero-Trust Data Policy**.
1. **No Individual Storage**: MindGuard reads text in-memory, counts the flags, aggregates the data, and throws the text away. 
2. **Reverse Identifiers Prevented**: Clients only ever see "5 Severe Crisis Flags in the last 30 minutes"—they never receive the raw text that triggered it, making anonymous witch-hunts impossible.
3. **Not a Diagnostic**: Hardcoded disclaimers in every API response explicitly define the tool as a *triage screening engine*, dodging heavy medical regulatory liabilities.
