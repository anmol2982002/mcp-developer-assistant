"""
Locust Load Testing Configuration

Load testing for MCP Developer Assistant endpoints.

Usage:
    locust -f locustfile.py --host=http://localhost:8001

Web UI will be available at http://localhost:8089
"""

from locust import HttpUser, task, between, events
import json
import random
import time


class MCPProxyUser(HttpUser):
    """Simulates users accessing MCP through the proxy."""
    
    wait_time = between(0.5, 2)  # Wait 0.5-2 seconds between requests
    
    def on_start(self):
        """Called when a simulated user starts."""
        # Get a test token (in real scenario, this would be OAuth)
        self.token = "test_bearer_token"
        self.headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json"
        }
        self.user_id = f"user_{random.randint(1000, 9999)}"
    
    @task(10)
    def health_check(self):
        """Health check endpoint - high frequency."""
        with self.client.get("/health", catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Health check failed: {response.status_code}")
    
    @task(5)
    def list_tools(self):
        """List available tools."""
        with self.client.get("/tools", headers=self.headers, catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            elif response.status_code == 401:
                response.success()  # Expected without valid token
            else:
                response.failure(f"List tools failed: {response.status_code}")
    
    @task(8)
    def read_file(self):
        """Simulate reading a file."""
        payload = {
            "tool_name": "read_file",
            "params": {
                "path": "README.md",
                "max_lines": 50
            },
            "user_id": self.user_id
        }
        with self.client.post(
            "/tool/read_file", 
            json=payload,
            headers=self.headers,
            catch_response=True
        ) as response:
            if response.status_code in [200, 401, 403]:
                response.success()
            else:
                response.failure(f"Read file failed: {response.status_code}")
    
    @task(3)
    def search_files(self):
        """Simulate searching files."""
        payload = {
            "tool_name": "search_files",
            "params": {
                "pattern": "def main",
                "directory": ".",
                "max_results": 10
            },
            "user_id": self.user_id
        }
        with self.client.post(
            "/tool/search_files",
            json=payload,
            headers=self.headers,
            catch_response=True
        ) as response:
            if response.status_code in [200, 401, 403]:
                response.success()
            else:
                response.failure(f"Search failed: {response.status_code}")
    
    @task(4)
    def git_status(self):
        """Check git status."""
        payload = {
            "tool_name": "git_status",
            "params": {},
            "user_id": self.user_id
        }
        with self.client.post(
            "/tool/git_status",
            json=payload,
            headers=self.headers,
            catch_response=True
        ) as response:
            if response.status_code in [200, 401, 403]:
                response.success()
            else:
                response.failure(f"Git status failed: {response.status_code}")
    
    @task(2)
    def ask_about_code(self):
        """Semantic code search."""
        queries = [
            "How does authentication work?",
            "Where is the rate limiter implemented?",
            "What tools are available?",
            "How are anomalies detected?",
        ]
        payload = {
            "tool_name": "ask_about_code",
            "params": {
                "query": random.choice(queries),
                "top_k": 3
            },
            "user_id": self.user_id
        }
        with self.client.post(
            "/tool/ask_about_code",
            json=payload,
            headers=self.headers,
            catch_response=True
        ) as response:
            if response.status_code in [200, 401, 403, 500]:  # LLM may not be available
                response.success()
            else:
                response.failure(f"Ask code failed: {response.status_code}")
    
    @task(1)
    def review_changes(self):
        """Code review request - expensive operation."""
        payload = {
            "tool_name": "review_changes",
            "params": {
                "ref": "HEAD~1"
            },
            "user_id": self.user_id
        }
        with self.client.post(
            "/tool/review_changes",
            json=payload,
            headers=self.headers,
            catch_response=True
        ) as response:
            if response.status_code in [200, 401, 403, 500]:
                response.success()
            else:
                response.failure(f"Review failed: {response.status_code}")


class MCPServerDirectUser(HttpUser):
    """Direct MCP server access (bypassing proxy) for testing."""
    
    wait_time = between(1, 3)
    
    # Override host for direct server access
    host = "http://localhost:8000"
    
    @task(10)
    def server_health(self):
        """Server health check."""
        self.client.get("/health")
    
    @task(5)
    def server_list_tools(self):
        """List tools directly."""
        self.client.get("/tools")


# Custom statistics collection
@events.request.add_listener
def on_request(request_type, name, response_time, response_length, **kwargs):
    """Track custom metrics."""
    # This could be extended to push metrics to Prometheus
    pass


@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    """Called when load test starts."""
    print("=" * 60)
    print("🚀 MCP Load Test Starting")
    print("=" * 60)


@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    """Called when load test stops."""
    print("=" * 60)
    print("🏁 MCP Load Test Complete")
    print("=" * 60)
