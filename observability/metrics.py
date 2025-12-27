"""
Prometheus Metrics

Exposes metrics for monitoring:
- Request counts and latencies
- ML confidence scores
- Anomaly detection
- Rate limiting
- Consent management
- Token operations
"""

from prometheus_client import Counter, Gauge, Histogram, Info

# Application info
APP_INFO = Info("mcp_app", "MCP Developer Assistant information")

# Request metrics
REQUESTS_TOTAL = Counter(
    "mcp_requests_total",
    "Total number of tool requests",
    ["tool"],
)

REQUESTS_DENIED = Counter(
    "mcp_requests_denied_total",
    "Number of denied requests",
    ["tool"],
)

REQUESTS_FAILED = Counter(
    "mcp_requests_failed_total",
    "Number of failed requests",
    ["tool"],
)

REQUEST_LATENCY = Histogram(
    "mcp_request_latency_seconds",
    "Request latency in seconds",
    ["tool"],
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
)

# Auth metrics
AUTH_ATTEMPTS = Counter(
    "mcp_auth_attempts_total",
    "Authentication attempts",
    ["status"],  # success, failed
)

ACTIVE_SESSIONS = Gauge(
    "mcp_active_sessions",
    "Number of active sessions",
)

# Token metrics (Phase 2)
TOKEN_REFRESHES = Counter(
    "mcp_token_refreshes_total",
    "Total token refresh operations",
)

TOKEN_INTROSPECTIONS = Counter(
    "mcp_token_introspections_total",
    "Total token introspection requests",
)

TOKEN_REVOCATIONS = Counter(
    "mcp_token_revocations_total",
    "Total token revocation operations",
)

# Intent checking metrics
INTENT_VIOLATIONS = Counter(
    "mcp_intent_violations_total",
    "Number of intent violations detected",
    ["tool"],
)

INTENT_CHECK_LATENCY = Histogram(
    "mcp_intent_check_latency_seconds",
    "Intent check latency in seconds",
    buckets=[0.1, 0.25, 0.5, 1.0, 2.0, 5.0],
)

INTENT_CONFIDENCE = Histogram(
    "mcp_intent_confidence",
    "Intent check confidence scores",
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)

# Anomaly detection metrics
ANOMALIES_DETECTED = Counter(
    "mcp_anomalies_detected_total",
    "Number of anomalies detected",
    ["user"],
)

ANOMALY_SCORE = Histogram(
    "mcp_anomaly_score",
    "Anomaly detection scores",
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)

# Rate limiting metrics (Phase 2)
RATE_LIMIT_HITS = Counter(
    "mcp_rate_limit_hits_total",
    "Total rate limit hits",
    ["user", "tier"],
)

RATE_LIMIT_REMAINING = Gauge(
    "mcp_rate_limit_remaining",
    "Remaining requests in current window",
    ["user"],
)

RATE_LIMIT_WINDOW = Histogram(
    "mcp_rate_limit_window_usage",
    "Rate limit window usage percentage",
    buckets=[0.1, 0.25, 0.5, 0.75, 0.9, 1.0],
)

# Consent metrics (Phase 2)
CONSENT_GRANTS = Counter(
    "mcp_consent_grants_total",
    "Total consent grants",
    ["client"],
)

CONSENT_REVOCATIONS = Counter(
    "mcp_consent_revocations_total",
    "Total consent revocations",
    ["client"],
)

CONSENT_CHECKS = Counter(
    "mcp_consent_checks_total",
    "Total consent checks",
    ["result"],  # granted, denied
)

# Embedding metrics
EMBEDDING_CACHE_SIZE = Gauge(
    "mcp_embedding_cache_size",
    "Number of cached embeddings",
)

EMBEDDING_LATENCY = Histogram(
    "mcp_embedding_latency_seconds",
    "Embedding computation latency",
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0],
)

# LLM metrics
LLM_CALLS = Counter(
    "mcp_llm_calls_total",
    "Number of LLM API calls",
    ["provider", "model"],
)

LLM_LATENCY = Histogram(
    "mcp_llm_latency_seconds",
    "LLM API call latency",
    ["provider"],
    buckets=[0.5, 1.0, 2.0, 5.0, 10.0, 30.0],
)

LLM_TOKENS = Counter(
    "mcp_llm_tokens_total",
    "Total LLM tokens used",
    ["provider", "type"],  # type: input, output
)

# MCP Client metrics (proxy -> server)
MCP_CLIENT_REQUESTS = Counter(
    "mcp_client_requests_total",
    "MCP client requests to server",
    ["tool", "status"],
)

MCP_CLIENT_LATENCY = Histogram(
    "mcp_client_latency_ms",
    "MCP client request latency in milliseconds",
    ["tool"],
    buckets=[10, 50, 100, 250, 500, 1000, 2500, 5000, 10000],
)

# Audit log metrics (Phase 2)
AUDIT_RECORDS = Counter(
    "mcp_audit_records_total",
    "Total audit log records",
    ["status"],  # success, denied, error
)

AUDIT_SANITIZATION = Counter(
    "mcp_audit_sanitization_total",
    "Audit log sanitization operations",
    ["type"],  # pii, secret, none
)

# -----------------------------------------------------------------------------
# Phase 3: ML Observability Metrics
# -----------------------------------------------------------------------------

# Anomaly detection metrics (enhanced)
ANOMALY_DETECTION_LATENCY = Histogram(
    "mcp_anomaly_detection_latency_seconds",
    "Anomaly detection processing latency",
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5],
)

ANOMALY_SCORE_BY_TYPE = Histogram(
    "mcp_anomaly_score_by_type",
    "Anomaly scores grouped by detection type",
    ["anomaly_type"],  # rapid_fire, night_access, sensitive, new_ip, etc.
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)

ANOMALY_EXPLANATIONS = Counter(
    "mcp_anomaly_explanations_total",
    "Anomaly explanations generated",
    ["risk_level"],  # LOW, MEDIUM, HIGH, CRITICAL
)

# Ensemble model metrics
ENSEMBLE_MODEL_SCORES = Histogram(
    "mcp_ensemble_model_scores",
    "Per-model scores from ensemble",
    ["model"],  # isolation_forest, lof, one_class_svm
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)

# Feature importance/contribution
FEATURE_CONTRIBUTION = Gauge(
    "mcp_feature_contribution",
    "Feature contribution to anomaly score",
    ["feature"],
)

# Intent cache metrics
INTENT_CACHE_SIZE = Gauge(
    "mcp_intent_cache_size",
    "Number of entries in intent cache",
)

INTENT_CACHE_HIT_RATE = Gauge(
    "mcp_intent_cache_hit_rate",
    "Intent cache hit rate (0-1)",
)

INTENT_CACHE_SEMANTIC_MATCHES = Counter(
    "mcp_intent_cache_semantic_matches_total",
    "Semantic cache matches (not exact)",
)

# Model version tracking
MODEL_VERSION_INFO = Info(
    "mcp_model_version",
    "Active anomaly detection model version",
)

MODEL_TRAINING_SAMPLES = Gauge(
    "mcp_model_training_samples",
    "Number of samples used to train active model",
)

# A/B testing metrics
AB_TEST_SELECTIONS = Counter(
    "mcp_ab_test_selections_total",
    "Model selections in A/B test",
    ["model_id"],
)

AB_TEST_PERFORMANCE = Histogram(
    "mcp_ab_test_performance",
    "Performance metrics per model in A/B test",
    ["model_id", "metric"],  # metric: anomaly_rate, latency, etc.
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)

# -----------------------------------------------------------------------------
# Phase 4: AI Code Tools Metrics
# -----------------------------------------------------------------------------

# Hybrid search metrics
HYBRID_SEARCH_LATENCY = Histogram(
    "mcp_hybrid_search_latency_seconds",
    "Hybrid search total latency",
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
)

BM25_SEARCH_SCORE = Histogram(
    "mcp_bm25_search_score",
    "BM25 keyword search scores",
    buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)

SEMANTIC_SEARCH_SCORE = Histogram(
    "mcp_semantic_search_score",
    "FAISS semantic search scores",
    buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)

RRF_FUSION_SCORE = Histogram(
    "mcp_rrf_fusion_score",
    "Reciprocal rank fusion scores",
    buckets=[0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.075, 0.1, 0.15, 0.2],
)

CODE_CHUNKS_INDEXED = Gauge(
    "mcp_code_chunks_indexed",
    "Total code chunks in search index",
)

FILES_INDEXED = Gauge(
    "mcp_files_indexed",
    "Total files in search index",
)

INDEX_BUILD_LATENCY = Histogram(
    "mcp_index_build_latency_seconds",
    "Index building latency",
    buckets=[1.0, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0],
)

# Output validation metrics
OUTPUT_VALIDATION_TOTAL = Counter(
    "mcp_output_validation_total",
    "Total output validation attempts",
    ["schema"],  # CodeReviewOutput, CodeSearchResult, etc.
)

OUTPUT_VALIDATION_FAILURES = Counter(
    "mcp_output_validation_failures_total",
    "Output validation failures",
    ["schema", "error_type"],  # error_type: json_decode, schema_error, etc.
)

# Secret detection metrics
SECRETS_DETECTED_TOTAL = Counter(
    "mcp_secrets_detected_total",
    "Secrets detected in LLM outputs",
    ["secret_type"],  # api_key, password, token, etc.
)

PII_DETECTED_TOTAL = Counter(
    "mcp_pii_detected_total",
    "PII detected in LLM outputs",
    ["pii_type"],  # email, phone, ssn, etc.
)

SECRETS_REDACTED_TOTAL = Counter(
    "mcp_secrets_redacted_total",
    "Total secrets redacted from outputs",
)

# Code summarization metrics
SUMMARIZATION_REQUESTS = Counter(
    "mcp_summarization_requests_total",
    "Summarization requests",
    ["type"],  # repo, diff, code
)

SUMMARIZATION_LATENCY = Histogram(
    "mcp_summarization_latency_seconds",
    "Summarization latency",
    buckets=[0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0],
)

# Query expansion metrics
QUERY_EXPANSIONS = Counter(
    "mcp_query_expansions_total",
    "Total query expansions performed",
)

QUERY_EXPANSION_TERMS_ADDED = Histogram(
    "mcp_query_expansion_terms_added",
    "Number of terms added by query expansion",
    buckets=[0, 1, 2, 3, 4, 5, 7, 10, 15, 20],
)

# -----------------------------------------------------------------------------
# Phase 5: Code Review Metrics
# -----------------------------------------------------------------------------

# Risk scoring metrics
CODE_REVIEW_RISK_SCORE = Histogram(
    "mcp_code_review_risk_score",
    "Risk score distribution for code reviews",
    buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)

CODE_REVIEW_ML_CONFIDENCE = Histogram(
    "mcp_code_review_ml_confidence",
    "ML model confidence for risk predictions",
    buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)

# Issue tracking
CODE_REVIEW_ISSUES = Counter(
    "mcp_code_review_issues_total",
    "Code review issues by severity and category",
    ["severity", "category"],  # HIGH/MEDIUM/LOW/INFO, security/performance/etc.
)

# Review routing
CODE_REVIEW_ROUTING = Counter(
    "mcp_code_review_routing_total",
    "Code review routing decisions",
    ["priority", "requires_security"],  # priority: critical/high/normal/low
)

CODE_REVIEW_TIME_ESTIMATE = Histogram(
    "mcp_code_review_time_estimate_minutes",
    "Estimated review time in minutes",
    buckets=[5, 10, 15, 20, 30, 45, 60, 90, 120],
)

# Feature extraction metrics
RISK_FEATURE_EXTRACTION_LATENCY = Histogram(
    "mcp_risk_feature_extraction_latency_seconds",
    "Risk feature extraction latency",
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5],
)

# Author familiarity tracking
AUTHOR_FAMILIARITY_SCORE = Histogram(
    "mcp_author_familiarity_score",
    "Author familiarity with codebase",
    buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)

class MetricsHelper:
    """Helper class for recording metrics."""

    def __init__(self):
        self.requests_total = REQUESTS_TOTAL
        self.requests_denied = REQUESTS_DENIED
        self.requests_failed = REQUESTS_FAILED
        self.request_latency = REQUEST_LATENCY
        self.intent_violations_total = INTENT_VIOLATIONS
        self.anomalies_detected_total = ANOMALIES_DETECTED
        self.mcp_client_requests_total = MCP_CLIENT_REQUESTS
        self.mcp_client_latency_ms = MCP_CLIENT_LATENCY

    def record_request(self, tool: str, latency: float, success: bool = True):
        """Record a tool request."""
        self.requests_total.labels(tool=tool).inc()
        self.request_latency.labels(tool=tool).observe(latency)

        if not success:
            self.requests_failed.labels(tool=tool).inc()

    def record_llm_call(
        self,
        provider: str,
        model: str,
        latency: float,
        input_tokens: int,
        output_tokens: int,
    ):
        """Record an LLM API call."""
        LLM_CALLS.labels(provider=provider, model=model).inc()
        LLM_LATENCY.labels(provider=provider).observe(latency)
        LLM_TOKENS.labels(provider=provider, type="input").inc(input_tokens)
        LLM_TOKENS.labels(provider=provider, type="output").inc(output_tokens)

    def record_rate_limit(self, user: str, tier: str, remaining: int):
        """Record rate limit status."""
        RATE_LIMIT_REMAINING.labels(user=user).set(remaining)
        if remaining == 0:
            RATE_LIMIT_HITS.labels(user=user, tier=tier).inc()

    def record_consent(self, client: str, granted: bool):
        """Record consent operation."""
        if granted:
            CONSENT_GRANTS.labels(client=client).inc()
        else:
            CONSENT_REVOCATIONS.labels(client=client).inc()

    def record_token_operation(self, operation: str):
        """Record token operation."""
        if operation == "refresh":
            TOKEN_REFRESHES.inc()
        elif operation == "introspect":
            TOKEN_INTROSPECTIONS.inc()
        elif operation == "revoke":
            TOKEN_REVOCATIONS.inc()

    # -------------------------------------------------------------------------
    # Phase 3: ML Metrics Recording
    # -------------------------------------------------------------------------

    def record_anomaly_detection(
        self,
        score: float,
        latency: float,
        is_anomalous: bool,
        anomaly_type: str = "unknown",
        risk_level: str = "LOW",
        model_scores: dict = None,
    ):
        """Record anomaly detection result."""
        ANOMALY_SCORE.observe(score)
        ANOMALY_DETECTION_LATENCY.observe(latency)
        
        if is_anomalous:
            ANOMALY_SCORE_BY_TYPE.labels(anomaly_type=anomaly_type).observe(score)
            ANOMALY_EXPLANATIONS.labels(risk_level=risk_level).inc()
        
        # Record per-model scores
        if model_scores:
            for model_name, model_score in model_scores.items():
                ENSEMBLE_MODEL_SCORES.labels(model=model_name).observe(model_score)

    def record_feature_contributions(self, contributions: dict):
        """Record feature contributions to anomaly score."""
        for feature_name, contribution in contributions.items():
            FEATURE_CONTRIBUTION.labels(feature=feature_name).set(contribution)

    def record_intent_cache_stats(self, size: int, hit_rate: float, is_semantic: bool = False):
        """Record intent cache statistics."""
        INTENT_CACHE_SIZE.set(size)
        INTENT_CACHE_HIT_RATE.set(hit_rate)
        
        if is_semantic:
            INTENT_CACHE_SEMANTIC_MATCHES.inc()

    def set_model_version(self, version: str, model_type: str, training_samples: int):
        """Set active model version info."""
        MODEL_VERSION_INFO.info({
            "version": version,
            "model_type": model_type,
        })
        MODEL_TRAINING_SAMPLES.set(training_samples)

    def record_ab_test_selection(self, model_id: str):
        """Record model selection in A/B test."""
        AB_TEST_SELECTIONS.labels(model_id=model_id).inc()

    def record_ab_test_performance(self, model_id: str, metric_name: str, value: float):
        """Record performance metric for A/B testing."""
        AB_TEST_PERFORMANCE.labels(model_id=model_id, metric=metric_name).observe(value)

    # -------------------------------------------------------------------------
    # Phase 5: Code Review Metrics Recording
    # -------------------------------------------------------------------------

    def record_code_review(
        self,
        risk_score: float,
        ml_confidence: float = None,
        issues: list = None,
        priority: str = "normal",
        requires_security: bool = False,
        estimated_time: int = None,
        author_familiarity: float = None,
    ):
        """
        Record code review metrics.
        
        Args:
            risk_score: ML-predicted risk score (0-1)
            ml_confidence: Model confidence (0-1)
            issues: List of issues (each with severity and category)
            priority: Review priority (critical, high, normal, low)
            requires_security: Whether security review required
            estimated_time: Estimated review time in minutes
            author_familiarity: Author familiarity score (0-1)
        """
        # Record risk score
        CODE_REVIEW_RISK_SCORE.observe(risk_score)
        
        # Record confidence if available
        if ml_confidence is not None:
            CODE_REVIEW_ML_CONFIDENCE.observe(ml_confidence)
        
        # Record issues by severity and category
        if issues:
            for issue in issues:
                severity = issue.get("severity", "INFO")
                category = issue.get("category", "other")
                CODE_REVIEW_ISSUES.labels(severity=severity, category=category).inc()
        
        # Record routing
        CODE_REVIEW_ROUTING.labels(
            priority=priority,
            requires_security=str(requires_security).lower(),
        ).inc()
        
        # Record estimated time
        if estimated_time is not None:
            CODE_REVIEW_TIME_ESTIMATE.observe(estimated_time)
        
        # Record author familiarity
        if author_familiarity is not None:
            AUTHOR_FAMILIARITY_SCORE.observe(author_familiarity)

    def record_risk_feature_extraction(self, latency: float):
        """Record risk feature extraction latency."""
        RISK_FEATURE_EXTRACTION_LATENCY.observe(latency)



# Singleton
metrics = MetricsHelper()

