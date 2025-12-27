# Developer Setup Guide

## Prerequisites

- **Python 3.10+** (3.11 recommended)
- **Git** for version control
- **Docker & Docker Compose** (optional, for containerized deployment)
- **(Optional)** CUDA-enabled GPU for faster embeddings

---

## Quick Start

### 1. Clone Repository

```bash
git clone https://github.com/yourusername/mcp-developer-assistant.git
cd mcp-developer-assistant
```

### 2. Create Virtual Environment

**Windows:**
```powershell
python -m venv venv
.\venv\Scripts\activate
```

**Linux/macOS:**
```bash
python -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt  # For development
```

### 4. Configure Environment

```bash
cp .env.example .env
```

Edit `.env` with your configuration:

```env
# Required for AI features
GROQ_API_KEY=your_groq_api_key
# OR
OPENAI_API_KEY=your_openai_key

# Optional
ENVIRONMENT=development
LOG_LEVEL=DEBUG
```

### 5. Initialize Database

```bash
python scripts/setup_db.py
```

### 6. Train ML Models (Optional)

```bash
python scripts/train_models.py
```

### 7. Start the Server

**Development mode:**
```bash
# Start MCP Server
python -m uvicorn server.mcp_server:app --reload --port 8000

# In another terminal - Start Proxy
python -m uvicorn proxy.auth_gateway:app --reload --port 8001
```

**Or use the combined script:**
```bash
python -m scripts.run_server
```

---

## Project Structure

```
mcp-developer-assistant/
├── proxy/              # OAuth + Security Layer
│   ├── auth_gateway.py     # Main proxy server
│   ├── intent_checker.py   # LLM intent validation
│   ├── anomaly_detector.py # ML anomaly detection
│   └── rate_limiter.py     # Sliding window limiter
│
├── server/             # MCP Server
│   ├── mcp_server.py       # FastAPI application
│   ├── tools/              # Tool implementations
│   └── policy_engine.py    # BAML policy enforcement
│
├── ai/                 # AI/ML Components
│   ├── embedding_manager.py    # Code embeddings
│   ├── hybrid_search.py        # BM25 + FAISS
│   ├── risk_scorer.py          # Risk assessment
│   └── llm_client.py           # LLM integration
│
├── observability/      # Monitoring
│   ├── metrics.py          # Prometheus metrics
│   ├── logging_config.py   # Structured logging
│   └── dashboards/         # Grafana dashboards
│
├── tests/              # Test suite
└── config/             # Configuration files
```

---

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# With coverage
pytest tests/ -v --cov=. --cov-report=html

# Specific test file
pytest tests/test_file_tools.py -v

# Validate project
python scripts/validate_project.py
```

---

## Docker Deployment

### Build and Run

```bash
docker-compose up --build
```

This starts:
- MCP Server on `:8000`
- Proxy Gateway on `:8001`
- PostgreSQL on `:5432`
- Prometheus on `:9090`
- Grafana on `:3000`

### Access Services

| Service | URL |
|---------|-----|
| MCP Server | http://localhost:8000 |
| Proxy | http://localhost:8001 |
| Grafana | http://localhost:3000 (admin/admin) |
| Prometheus | http://localhost:9090 |

---

## Load Testing

```bash
# Install locust
pip install locust

# Run load tests
locust -f locustfile.py --host=http://localhost:8001

# Open http://localhost:8089 for web UI
```

---

## Claude Desktop Integration

Add to your Claude Desktop config (`claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "developer-assistant": {
      "command": "python",
      "args": ["-m", "uvicorn", "server.mcp_server:app", "--port", "8000"],
      "cwd": "/path/to/mcp-developer-assistant"
    }
  }
}
```

---

## Troubleshooting

### Import Errors

Ensure you're in the project root and venv is activated:
```bash
cd mcp-developer-assistant
source venv/bin/activate  # or .\venv\Scripts\activate on Windows
```

### LLM Features Not Working

Check your API key in `.env`:
```bash
echo $GROQ_API_KEY  # Should show your key
```

### Database Issues

Reset the database:
```bash
rm -f data/mcp.db
python scripts/setup_db.py
```

### ML Model Issues

Retrain models:
```bash
python scripts/train_models.py --force
```

---

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Make your changes
4. Run tests: `pytest tests/ -v`
5. Submit a pull request

---

## Resources

- [MCP Protocol Documentation](https://modelcontextprotocol.io)
- [FastAPI Documentation](https://fastapi.tiangolo.com)
- [Prometheus Metrics](https://prometheus.io/docs/concepts/metric_types/)
