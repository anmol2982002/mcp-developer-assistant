<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11+-blue?style=for-the-badge&logo=python" alt="Python">
  <img src="https://img.shields.io/badge/FastAPI-0.104+-green?style=for-the-badge&logo=fastapi" alt="FastAPI">
  <img src="https://img.shields.io/badge/Docker-Ready-blue?style=for-the-badge&logo=docker" alt="Docker">
  <img src="https://img.shields.io/badge/ML-Powered-purple?style=for-the-badge&logo=pytorch" alt="ML">
  <img src="https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge" alt="License">
</p>

<h1 align="center">🤖 MCP Developer Assistant</h1>

<p align="center">
  <strong>A production-grade, AI-powered Model Context Protocol (MCP) server with enterprise security, behavioral anomaly detection, and intelligent code assistance.</strong>
</p>

<p align="center">
  <a href="#-key-features">Features</a> •
  <a href="#-architecture">Architecture</a> •
  <a href="#-quick-start">Quick Start</a> •
  <a href="#-tools-available">Tools</a> •
  <a href="#-security">Security</a> •
  <a href="#-observability">Observability</a>
</p>

---

## 🎯 Why This Project Stands Out

> **Built over 6 iterative phases**, this project demonstrates production-ready software engineering with **ML/AI integration**, **enterprise security patterns**, and **full observability** — the skills that matter for Staff/Senior ML Engineer roles.

| Skill Demonstrated | Implementation |
|-------------------|----------------|
| **ML Engineering** | Ensemble anomaly detection (Isolation Forest + LOF + One-Class SVM), SHAP explainability, model registry with A/B testing |
| **LLM Integration** | Groq/OpenAI clients, semantic caching (70% LLM call reduction), intent validation with LLM-as-Judge pattern |
| **Security** | OAuth 2.1 with PKCE, confused deputy prevention, sliding window rate limiting, audit logging with PII sanitization |
| **Backend Development** | FastAPI async services, SQLAlchemy ORM, structured logging, Prometheus metrics |
| **DevOps** | Multi-stage Docker builds, GitHub Actions CI/CD, Grafana dashboards, health checks |
| **Code Quality** | 18 test files, type hints, BAML policy engine, comprehensive documentation |

---

## 🚀 Key Features

### 🔧 Traditional Developer Tools
- **File Tools** — Read, search, and list files with access control
- **Git Tools** — Status, diff, log with enhanced parsing
- **Code Analysis** — AST-based function/class extraction

### 🧠 AI-Powered Tools
- **`ask_about_code`** — Semantic code Q&A with RAG (BM25 + FAISS hybrid search)
- **`summarize_repo`** — Generate intelligent project summaries
- **`summarize_diff`** — AI-powered PR/commit summaries
- **`review_changes`** — Automated code review with risk scoring

### 🛡️ Enterprise Security
- **OAuth 2.1 Proxy** — Token validation, refresh token rotation, PKCE support
- **Confused Deputy Prevention** — Per-client consent management
- **Intent Checking** — LLM validates requests match tool purpose
- **Behavioral Anomaly Detection** — ML learns normal patterns, flags suspicious access
- **Rate Limiting** — Sliding window with per-user quotas and burst handling

### 📊 Full Observability
- **Prometheus Metrics** — 50+ custom metrics including ML confidence scores
- **Grafana Dashboards** — Real-time anomaly detection, LLM cost tracking
- **Structured Logging** — JSON format with request tracing
- **Health Checks** — Kubernetes-ready endpoints

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     MCP CLIENT LAYER                             │
│          (Claude Desktop / VS Code / Custom Client)              │
└────────────────────────┬────────────────────────────────────────┘
                         │ JSON-RPC 2.0
┌────────────────────────▼────────────────────────────────────────┐
│              MCP PROXY (Security Gateway) :8001                  │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ 1. OAuth 2.1 Token Validation                              │ │
│  │ 2. Confused Deputy Prevention (Consent Check)              │ │
│  │ 3. Intent Checking (LLM-as-Judge) ⭐ ML                    │ │
│  │ 4. Behavioral Anomaly Detection (Ensemble) ⭐ ML           │ │
│  │ 5. Sliding Window Rate Limiting                            │ │
│  │ 6. Audit Logging with PII Sanitization                     │ │
│  └────────────────────────────────────────────────────────────┘ │
└────────────────────────┬────────────────────────────────────────┘
                         │ Authenticated Request
┌────────────────────────▼────────────────────────────────────────┐
│              MCP SERVER (Tool Execution) :8000                   │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ Policy Engine (BAML-based validation)                   │    │
│  │ ├── File Tools (read, search, list)                     │    │
│  │ ├── Git Tools (status, diff, log)                       │    │
│  │ ├── Code Tools (extract functions, analyze imports)     │    │
│  │ └── AI Tools (semantic search, summarize, review) ⭐    │    │
│  └─────────────────────────────────────────────────────────┘    │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ ML Intelligence Layer                                    │    │
│  │ ├── Hybrid Search (BM25 + FAISS embeddings)             │    │
│  │ ├── Risk Scorer (ML-based change risk prediction)       │    │
│  │ └── Output Validator (secret detection, PII filtering)  │    │
│  └─────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

---

## ⚡ Quick Start

### Prerequisites
- Python 3.10+ (3.11 recommended)
- Git
- (Optional) Docker for containerized deployment

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/mcp-developer-assistant.git
cd mcp-developer-assistant

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: .\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your GROQ_API_KEY or OPENAI_API_KEY

# Initialize database
python scripts/setup_db.py

# (Optional) Train ML models
python scripts/train_models.py
```

### Run the Server

```bash
# Start MCP Server
python -m uvicorn server.mcp_server:app --reload --port 8000

# In another terminal - Start Proxy
python -m uvicorn proxy.auth_gateway:app --reload --port 8001
```

### Validate Installation

```bash
# Run comprehensive project validation
python scripts/validate_project.py

# Run test suite
pytest tests/ -v --cov=. --cov-report=html
```

---

## 🐳 Docker Deployment

```bash
# Start all services
docker-compose up --build

# Access services:
# - MCP Server:  http://localhost:8000
# - Proxy:       http://localhost:8001  
# - Grafana:     http://localhost:3000 (admin/admin)
# - Prometheus:  http://localhost:9090
```

---

## 🛠️ Tools Available

| Tool | Description | AI-Powered |
|------|-------------|:----------:|
| `read_file` | Read file with line range support | |
| `search_files` | Regex/pattern search across files | |
| `list_directory` | List directory contents | |
| `git_status` | Repository status | |
| `git_diff` | Show changes between refs | |
| `git_log` | Commit history | |
| `extract_functions` | AST-based function extraction | |
| `ask_about_code` | Semantic code Q&A | ✅ |
| `summarize_repo` | Project overview generation | ✅ |
| `summarize_diff` | Change summary generation | ✅ |
| `review_changes` | Automated code review | ✅ |

---

## 🔐 Security Features

### OAuth 2.1 Implementation
- PKCE support for public clients
- Refresh token rotation
- Token introspection endpoint
- Configurable scopes per client

### ML-Based Intent Checking
```python
# Detects when tools are misused (e.g., read_file for secret exfiltration)
intent_result = await intent_checker.validate_intent(
    tool_name="read_file",
    params={"path": ".env.production"},
    user_intent="Review configuration"
)
# Returns: {is_valid: False, confidence: 0.95, reason: "Accessing secrets file"}
```

### Behavioral Anomaly Detection
- **Ensemble Model**: Isolation Forest + Local Outlier Factor + One-Class SVM
- **10+ Behavioral Features**: Request rate, tool sequences, time patterns, IP changes
- **SHAP Explainability**: Human-readable explanations for anomalies
- **Real-time Updates**: Continuous learning from audit logs

---

## 📊 Observability

### Prometheus Metrics (50+ metrics)
```
mcp_requests_total{tool="read_file"}
mcp_anomaly_score_bucket{le="0.5"}
mcp_llm_tokens_total{provider="groq", model="llama-3.3-70b"}
mcp_intent_cache_hit_rate
```

### Grafana Dashboards
- **ML Anomaly Detection** — Real-time anomaly visualization, feature contributions
- **Request Latency** — p50/p95/p99, error rates, rate limiting
- **LLM Cost Tracking** — Token usage, estimated costs, cache savings

---

## 🧪 Testing

```bash
# Run all tests with coverage
pytest tests/ -v --cov=. --cov-report=html

# Run specific test categories
pytest tests/test_file_tools.py -v
pytest tests/test_anomaly_detector.py -v
pytest tests/test_code_review_phase5.py -v

# Load testing
pip install locust
locust -f locustfile.py --host=http://localhost:8001
```

**Test Coverage**: 18 test files covering all phases

---

## 📁 Project Structure

```
mcp-developer-assistant/
├── proxy/                    # OAuth + Security Gateway
│   ├── auth_gateway.py           # Main proxy (FastAPI)
│   ├── intent_checker.py         # LLM intent validation
│   ├── anomaly_detector.py       # Ensemble ML detection
│   ├── rate_limiter.py           # Sliding window limiter
│   └── consent_db.py             # Consent management
│
├── server/                   # MCP Server
│   ├── mcp_server.py             # Tool dispatcher
│   ├── tools/                    # Tool implementations
│   └── policy_engine.py          # BAML policies
│
├── ai/                       # ML/AI Components
│   ├── embedding_manager.py      # Sentence transformers
│   ├── hybrid_search.py          # BM25 + FAISS
│   ├── risk_scorer.py            # Risk prediction
│   ├── model_trainer.py          # Training pipeline
│   └── shap_explainer.py         # Explainability
│
├── observability/            # Monitoring
│   ├── metrics.py                # Prometheus (608 lines)
│   ├── dashboards/               # Grafana JSON
│   └── health_check.py           # K8s-ready checks
│
├── tests/                    # 18 test files
├── docs/                     # Documentation
├── .github/workflows/        # CI/CD
├── docker-compose.yml        # Full stack deployment
└── locustfile.py             # Load testing
```

---

## 🛤️ Development Roadmap

| Phase | Status | Description |
|-------|--------|-------------|
| Phase 0 | ✅ Complete | Project structure (55+ files) |
| Phase 1 | ✅ Complete | MVP Core (File/Git/Code tools) |
| Phase 2 | ✅ Complete | Security & Proxy (OAuth 2.1) |
| Phase 3 | ✅ Complete | ML Security (Anomaly Detection) |
| Phase 4 | ✅ Complete | AI Tools (Semantic Search) |
| Phase 5 | ✅ Complete | Code Review (Risk Scoring) |
| Phase 6 | ✅ Complete | Production Ready (CI/CD, Docs) |

---

## 📄 Documentation

- [API Documentation](docs/API.md)
- [Developer Setup Guide](docs/DEVELOPER_SETUP.md)
- [Architecture Decisions](docs/ARCHITECTURE.md)

---

## 🤝 Contributing

Contributions are welcome! Please read the [Contributing Guide](CONTRIBUTING.md) first.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Run tests (`pytest tests/ -v`)
4. Commit your changes (`git commit -m 'Add amazing feature'`)
5. Push to branch (`git push origin feature/amazing-feature`)
6. Open a Pull Request

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👨‍💻 Author

**Built with ❤️ as a demonstration of production-grade ML engineering**

*This project showcases the complete software development lifecycle: from architecture design through implementation, testing, and deployment — demonstrating the skills needed for Staff/Senior ML/Backend Engineering roles.*

---

<p align="center">
  <strong>If you found this useful, please ⭐ the repository!</strong>
</p>
