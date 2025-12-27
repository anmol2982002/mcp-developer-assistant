# End-to-End Usage Guide

Complete guide to using MCP Developer Assistant from installation to production deployment.

---

## Table of Contents

1. [Installation](#1-installation)
2. [Configuration](#2-configuration)
3. [Starting the Server](#3-starting-the-server)
4. [Using the Tools](#4-using-the-tools)
5. [Claude Desktop Integration](#5-claude-desktop-integration)
6. [Monitoring & Observability](#6-monitoring--observability)
7. [Production Deployment](#7-production-deployment)
8. [Troubleshooting](#8-troubleshooting)

---

## 1. Installation

### Step 1: Clone and Setup

```bash
# Clone repository
git clone https://github.com/yourusername/mcp-developer-assistant.git
cd mcp-developer-assistant

# Create virtual environment
python -m venv venv

# Activate (Windows)
.\venv\Scripts\activate

# Activate (Linux/macOS)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Configure Environment

```bash
# Copy example environment file
cp .env.example .env
```

Edit `.env` with your settings:

```env
# ============================================
# REQUIRED: At least one LLM provider
# ============================================
GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxx
# OR
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxx

# ============================================
# OPTIONAL: Customize behavior
# ============================================
ENVIRONMENT=development
LOG_LEVEL=INFO
DATABASE_URL=sqlite:///data/mcp.db

# OAuth settings (for proxy)
JWT_SECRET=your-secret-key-min-32-chars
JWT_ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30

# Rate limiting
RATE_LIMIT_DEFAULT=60
RATE_LIMIT_BURST=100
```

### Step 3: Initialize Database

```bash
python scripts/setup_db.py
```

### Step 4: Train ML Models (Optional but Recommended)

```bash
python scripts/train_models.py
```

This trains:
- Anomaly detection ensemble model
- Intent classification baseline
- Risk prediction model

---

## 2. Configuration

### Tool Intent Configuration

Edit `config/tool_intents.yaml` to customize tool permissions:

```yaml
read_file:
  intent: "Read non-sensitive code/documentation files"
  allowed_patterns:
    - "*.py"
    - "*.md"
    - "*.json"
  forbidden_patterns:
    - ".env*"
    - "*secret*"
    - "credentials*"
```

### Rate Limit Tiers

Edit `proxy/config.py`:

```python
RATE_LIMIT_TIERS = {
    "free": {"requests_per_minute": 10, "burst": 20},
    "standard": {"requests_per_minute": 60, "burst": 100},
    "premium": {"requests_per_minute": 300, "burst": 500},
}
```

---

## 3. Starting the Server

### Development Mode

**Terminal 1 - MCP Server:**
```bash
python -m uvicorn server.mcp_server:app --reload --port 8000
```

**Terminal 2 - Proxy Gateway:**
```bash
python -m uvicorn proxy.auth_gateway:app --reload --port 8001
```

### Production Mode (Docker)

```bash
docker-compose up -d
```

### Verify Installation

```bash
# Check MCP Server health
curl http://localhost:8000/health

# Check Proxy health
curl http://localhost:8001/health

# List available tools
curl http://localhost:8000/tools | jq .
```

---

## 4. Using the Tools

### Direct API Usage

**Read a File:**
```bash
curl -X POST http://localhost:8000/tool/read_file \
  -H "Content-Type: application/json" \
  -d '{
    "tool_name": "read_file",
    "params": {"path": "README.md", "max_lines": 50}
  }'
```

**Search Files:**
```bash
curl -X POST http://localhost:8000/tool/search_files \
  -H "Content-Type: application/json" \
  -d '{
    "tool_name": "search_files",
    "params": {"pattern": "def main", "directory": "."}
  }'
```

**Git Status:**
```bash
curl -X POST http://localhost:8000/tool/git_status \
  -H "Content-Type: application/json" \
  -d '{"tool_name": "git_status", "params": {}}'
```

**Semantic Code Search (AI):**
```bash
curl -X POST http://localhost:8000/tool/ask_about_code \
  -H "Content-Type: application/json" \
  -d '{
    "tool_name": "ask_about_code",
    "params": {
      "query": "How does authentication work?",
      "top_k": 5
    }
  }'
```

**Code Review (AI):**
```bash
curl -X POST http://localhost:8000/tool/review_changes \
  -H "Content-Type: application/json" \
  -d '{
    "tool_name": "review_changes",
    "params": {"ref": "HEAD~1"}
  }'
```

### Using Through Proxy (Authenticated)

```bash
# Get token (in real scenario, use OAuth flow)
TOKEN="your_jwt_token"

curl -X POST http://localhost:8001/tool_call \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "tool_name": "read_file",
    "params": {"path": "README.md"},
    "user_stated_intent": "Reviewing documentation"
  }'
```

---

## 5. Claude Desktop Integration

### Step 1: Locate Config File

- **Windows:** `%APPDATA%\Claude\claude_desktop_config.json`
- **macOS:** `~/Library/Application Support/Claude/claude_desktop_config.json`
- **Linux:** `~/.config/claude/claude_desktop_config.json`

### Step 2: Add MCP Server Configuration

```json
{
  "mcpServers": {
    "developer-assistant": {
      "command": "python",
      "args": [
        "-m", "uvicorn", 
        "server.mcp_server:app", 
        "--port", "8000"
      ],
      "cwd": "C:/path/to/mcp-developer-assistant",
      "env": {
        "GROQ_API_KEY": "your_groq_api_key"
      }
    }
  }
}
```

### Step 3: Restart Claude Desktop

After saving, restart Claude Desktop. You should see the tools available in the interface.

### Step 4: Use Tools in Claude

In Claude Desktop, you can now ask:
- "Read the README.md file"
- "Search for all functions that handle authentication"
- "What does the rate limiter do?"
- "Review the latest commit changes"

---

## 6. Monitoring & Observability

### Access Grafana

1. Start with Docker: `docker-compose up -d`
2. Open http://localhost:3000
3. Login: admin/admin
4. Import dashboards from `observability/dashboards/`

### Available Dashboards

| Dashboard | Description |
|-----------|-------------|
| ML Anomaly Detection | Anomaly scores, feature contributions, risk levels |
| Request Latency | p50/p95/p99 latency, error rates, rate limiting |
| LLM Cost Tracking | Token usage, cost estimation, cache savings |

### Prometheus Metrics

Access http://localhost:9090 for direct metric queries:

```promql
# Request rate by tool
rate(mcp_requests_total[5m])

# Anomaly detection rate
rate(mcp_anomalies_detected_total[5m])

# LLM token usage
sum(increase(mcp_llm_tokens_total[1h]))
```

### Structured Logs

Logs are JSON formatted for easy parsing:

```bash
# View logs
docker-compose logs -f mcp-server

# Parse with jq
docker-compose logs mcp-server 2>&1 | jq '.message'
```

---

## 7. Production Deployment

### Docker Compose (Recommended)

```bash
# Start all services
docker-compose up -d

# View status
docker-compose ps

# View logs
docker-compose logs -f

# Scale services
docker-compose up -d --scale mcp-server=3

# Stop services
docker-compose down
```

### Environment Variables for Production

```env
ENVIRONMENT=production
LOG_LEVEL=WARNING

# Strong secrets
JWT_SECRET=generate-a-strong-64-char-secret

# Database (use PostgreSQL in production)
DATABASE_URL=postgresql://user:pass@db:5432/mcp

# Rate limiting (stricter)
RATE_LIMIT_DEFAULT=30
RATE_LIMIT_BURST=50
```

### Health Checks

Both services expose `/health` endpoints:

```bash
# Check all services
curl http://localhost:8000/health  # MCP Server
curl http://localhost:8001/health  # Proxy
```

### Load Testing

```bash
# Install locust
pip install locust

# Run load tests
locust -f locustfile.py --host=http://localhost:8001

# Open web UI at http://localhost:8089
# Configure: 100 users, spawn rate 10/s
```

---

## 8. Troubleshooting

### Common Issues

**Port Already in Use:**
```bash
# Find process using port
netstat -ano | findstr :8000

# Kill process (Windows)
taskkill /PID <pid> /F
```

**Import Errors:**
```bash
# Ensure you're in project root with venv activated
cd mcp-developer-assistant
.\venv\Scripts\activate
python -c "import server.mcp_server"  # Should not error
```

**LLM Not Working:**
```bash
# Verify API key
echo $GROQ_API_KEY  # Should show your key

# Test LLM client
python -c "from ai.llm_client import LLMClient; print('OK')"
```

**Database Issues:**
```bash
# Reset database
rm -f data/mcp.db
python scripts/setup_db.py
```

**ML Models Not Loading:**
```bash
# Retrain models
python scripts/train_models.py --force
```

### Validation

Run the comprehensive validation script:

```bash
python scripts/validate_project.py -v
```

This checks:
- All imports work
- Required classes exist
- File structure is correct
- Configuration is valid

### Getting Help

1. Check logs: `docker-compose logs -f`
2. Run validation: `python scripts/validate_project.py -v`
3. Run tests: `pytest tests/ -v`
4. Open an issue on GitHub

---

## Quick Reference

| Command | Description |
|---------|-------------|
| `python scripts/setup_db.py` | Initialize database |
| `python scripts/train_models.py` | Train ML models |
| `python scripts/validate_project.py` | Validate installation |
| `pytest tests/ -v` | Run test suite |
| `docker-compose up -d` | Start all services |
| `locust -f locustfile.py` | Run load tests |

---

*For more details, see [API Documentation](API.md) and [Developer Setup](DEVELOPER_SETUP.md).*
