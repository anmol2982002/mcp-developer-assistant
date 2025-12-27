# API Documentation

## Overview

The MCP Developer Assistant exposes two main services:
- **MCP Server** (`:8000`) - Core tool execution
- **Proxy Gateway** (`:8001`) - Authentication, intent checking, anomaly detection

---

## MCP Server Endpoints

### Health Check

```
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "version": "0.1.0",
  "tools_loaded": 10
}
```

### List Tools

```
GET /tools
```

**Response:**
```json
{
  "tools": [
    {
      "name": "read_file",
      "description": "Read a file with optional line range",
      "inputSchema": {...}
    }
  ]
}
```

### Execute Tool

```
POST /tool/{tool_name}
Content-Type: application/json
```

**Request Body:**
```json
{
  "tool_name": "read_file",
  "params": {
    "path": "README.md",
    "max_lines": 100
  },
  "user_id": "optional_user_id",
  "client_id": "optional_client_id"
}
```

**Success Response:**
```json
{
  "success": true,
  "result": {
    "content": "# File content...",
    "lines": 50
  }
}
```

**Error Response:**
```json
{
  "success": false,
  "error": "File not found: path/to/file"
}
```

---

## Available Tools

### File Tools

| Tool | Description |
|------|-------------|
| `read_file` | Read file contents with line range support |
| `search_files` | Search for patterns across files |
| `list_directory` | List directory contents |

### Git Tools

| Tool | Description |
|------|-------------|
| `git_status` | Get repository status |
| `git_diff` | Show changes between commits |
| `git_log` | View commit history |

### Code Tools

| Tool | Description |
|------|-------------|
| `extract_functions` | Extract function definitions |
| `analyze_imports` | Analyze import statements |

### AI Tools

| Tool | Description |
|------|-------------|
| `ask_about_code` | Semantic code Q&A |
| `summarize_repo` | Generate repository summary |
| `summarize_diff` | Summarize code changes |
| `review_changes` | AI-powered code review |

---

## Proxy Gateway Endpoints

### Health Check

```
GET /health
```

### Authenticated Tool Call

```
POST /tool_call
Authorization: Bearer <token>
Content-Type: application/json
```

**Request Body:**
```json
{
  "tool_name": "read_file",
  "params": {"path": "README.md"},
  "user_stated_intent": "Reviewing documentation"
}
```

### Token Introspection

```
POST /oauth/introspect
Content-Type: application/x-www-form-urlencoded

token=<access_token>
```

### Consent Management

```
POST /consent/grant
Authorization: Bearer <token>
Content-Type: application/json

{
  "client_id": "claude-desktop",
  "scopes": ["read_file", "git_status"]
}
```

---

## Error Codes

| Code | Description |
|------|-------------|
| 400 | Bad Request - Invalid parameters |
| 401 | Unauthorized - Invalid/missing token |
| 403 | Forbidden - Insufficient permissions or consent |
| 404 | Not Found - Tool or resource not found |
| 429 | Too Many Requests - Rate limit exceeded |
| 500 | Internal Server Error |

---

## Rate Limiting

The proxy implements sliding window rate limiting:

| Tier | Requests/minute | Burst |
|------|-----------------|-------|
| Free | 10 | 20 |
| Standard | 60 | 100 |
| Premium | 300 | 500 |

Rate limit headers are included in responses:
```
X-RateLimit-Limit: 60
X-RateLimit-Remaining: 45
X-RateLimit-Reset: 1703678400
```

---

## Prometheus Metrics

Available at `/metrics`:

- `mcp_requests_total` - Total requests by tool
- `mcp_request_latency_seconds` - Request latency histogram
- `mcp_anomaly_score` - Anomaly detection scores
- `mcp_intent_violations_total` - Intent check failures
- `mcp_llm_tokens_total` - LLM token usage
