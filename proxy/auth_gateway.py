"""
MCP Proxy - Authorization Gateway

Main FastAPI application for the proxy server that handles:
- OAuth 2.1 token validation with PKCE
- Refresh token rotation
- Confused deputy prevention
- Intent checking (LLM-as-Judge)
- Behavioral anomaly detection
- Sliding window rate limiting
- Request forwarding to MCP server
- Audit log API
"""

import json
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from typing import List, Optional

from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from database.audit_log import AuditLogDB, AuditQueryFilters, Pagination
from database.schema import get_db, init_db
from observability.logging_config import get_logger
from observability.metrics import metrics, ANOMALY_SCORE, INTENT_CONFIDENCE, AUTH_ATTEMPTS
from proxy.anomaly_detector import BehavioralAnomalyDetector, ToolRequest as AnomalyRequest
from proxy.config import ProxyConfig, get_proxy_config
from proxy.consent_db import ConsentDB
from proxy.intent_checker import IntentChecker
from proxy.mcp_client import mcp_client
from proxy.oauth_validator import OAuthValidator, oauth_validator, TokenResponse
from proxy.rate_limiter import rate_limiter
from proxy.refresh_token_store import RefreshTokenStore

logger = get_logger(__name__)


# -----------------------------------------------------------------------------
# Request/Response Models
# -----------------------------------------------------------------------------


class ToolRequest(BaseModel):
    """Tool request from MCP client."""

    tool_name: str
    params: dict
    user_stated_intent: Optional[str] = None
    client_id: Optional[str] = None


class ToolResponse(BaseModel):
    """Tool response to MCP client."""

    success: bool
    result: Optional[dict] = None
    error: Optional[str] = None
    duration_ms: Optional[int] = None


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    version: str = "0.2.0"
    components: Optional[dict] = None


class TokenRequestForm(BaseModel):
    """OAuth token request."""
    
    grant_type: str  # authorization_code, refresh_token
    code: Optional[str] = None
    code_verifier: Optional[str] = None
    refresh_token: Optional[str] = None
    client_id: str
    redirect_uri: Optional[str] = None


class ConsentGrantRequest(BaseModel):
    """Consent grant request."""
    
    client_id: str
    scopes: List[str]
    expires_days: Optional[int] = None


class ConsentRevokeRequest(BaseModel):
    """Consent revoke request."""
    
    client_id: str


class AuditQueryRequest(BaseModel):
    """Audit log query request."""
    
    user_id: Optional[str] = None
    client_id: Optional[str] = None
    tool_name: Optional[str] = None
    status: Optional[str] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    anomaly_detected: Optional[bool] = None
    page: int = 1
    page_size: int = 50


# -----------------------------------------------------------------------------
# Application State
# -----------------------------------------------------------------------------


class AppState:
    """Application state container for initialized components."""

    def __init__(self):
        self.oauth_validator: OAuthValidator = oauth_validator
        self.intent_checker: IntentChecker = IntentChecker()
        self.anomaly_detector: BehavioralAnomalyDetector = BehavioralAnomalyDetector()
        self.initialized: bool = False

    async def initialize(self):
        """Initialize all components."""
        if self.initialized:
            return

        # Initialize database
        await init_db()

        # Load anomaly model if exists
        config = get_proxy_config()
        if config.anomaly_model_path:
            self.anomaly_detector = BehavioralAnomalyDetector(
                model_path=config.anomaly_model_path
            )

        # Check MCP server health
        mcp_healthy = await mcp_client.health_check()
        if not mcp_healthy:
            logger.warning("mcp_server_not_available", msg="MCP server not reachable")

        self.initialized = True
        logger.info("proxy_initialized", components=["oauth", "intent", "anomaly", "mcp_client"])

    async def shutdown(self):
        """Cleanup on shutdown."""
        await mcp_client.close()
        logger.info("proxy_shutdown_complete")


app_state = AppState()


# -----------------------------------------------------------------------------
# Application Lifespan
# -----------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler for startup/shutdown."""
    # Startup
    logger.info("Starting MCP Proxy server...")
    await app_state.initialize()

    yield

    # Shutdown
    logger.info("Shutting down MCP Proxy server...")
    await app_state.shutdown()


# -----------------------------------------------------------------------------
# FastAPI Application
# -----------------------------------------------------------------------------


app = FastAPI(
    title="MCP Developer Assistant - Proxy",
    description="Authorization gateway with OAuth 2.1, intent checking, and anomaly detection",
    version="0.2.0",
    lifespan=lifespan,
)


# CORS middleware
config = get_proxy_config()
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------


def get_client_ip(request: Request) -> str:
    """Extract client IP from request."""
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def generate_request_id() -> str:
    """Generate unique request ID."""
    return str(uuid.uuid4())


# -----------------------------------------------------------------------------
# Health Endpoint
# -----------------------------------------------------------------------------


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint with component status."""
    mcp_healthy = await mcp_client.health_check()

    return HealthResponse(
        status="healthy" if app_state.initialized else "initializing",
        components={
            "oauth": True,
            "intent_checker": True,
            "anomaly_detector": app_state.anomaly_detector.model is not None,
            "mcp_server": mcp_healthy,
        },
    )


# -----------------------------------------------------------------------------
# OAuth 2.1 Endpoints
# -----------------------------------------------------------------------------


@app.post("/oauth/token", response_model=TokenResponse)
async def oauth_token(
    request: TokenRequestForm,
    http_request: Request,
):
    """
    OAuth 2.1 token endpoint.
    
    Supports:
    - authorization_code: Exchange code for tokens (with PKCE)
    - refresh_token: Refresh access token
    """
    config = get_proxy_config()
    
    if request.grant_type == "authorization_code":
        # PKCE authorization code flow
        if not request.code or not request.code_verifier or not request.redirect_uri:
            raise HTTPException(
                status_code=400,
                detail="Missing required parameters: code, code_verifier, redirect_uri",
            )
        
        async with get_db() as session:
            token_response = await app_state.oauth_validator.exchange_authorization_code(
                code=request.code,
                code_verifier=request.code_verifier,
                client_id=request.client_id,
                redirect_uri=request.redirect_uri,
                session=session,
            )
            
            if not token_response:
                raise HTTPException(status_code=400, detail="Invalid authorization code")
            
            # Create refresh token if rotation is enabled
            if config.refresh_token_rotation:
                token_store = RefreshTokenStore(session)
                refresh_token, _ = await token_store.create_refresh_token(
                    user_id=token_response.access_token,  # Decode to get user_id
                    client_id=request.client_id,
                    scopes=token_response.scope.split() if token_response.scope else [],
                    expiry_days=config.refresh_token_expiry_days,
                )
                token_response.refresh_token = refresh_token
            
            return token_response
    
    elif request.grant_type == "refresh_token":
        # Refresh token flow
        if not request.refresh_token:
            raise HTTPException(status_code=400, detail="Missing refresh_token")
        
        async with get_db() as session:
            token_store = RefreshTokenStore(session)
            
            # Rotate refresh token
            result = await token_store.rotate_refresh_token(
                old_token=request.refresh_token,
                expiry_days=config.refresh_token_expiry_days,
            )
            
            if not result:
                raise HTTPException(status_code=400, detail="Invalid refresh token")
            
            new_refresh_token, token_data = result
            
            # Create new access token
            access_token = app_state.oauth_validator.create_token(
                user_id=token_data.user_id,
                client_id=token_data.client_id,
                scopes=token_data.scopes,
            )
            
            return TokenResponse(
                access_token=access_token,
                token_type="Bearer",
                expires_in=config.jwt_expiration_hours * 3600,
                refresh_token=new_refresh_token,
                scope=" ".join(token_data.scopes),
            )
    
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported grant_type: {request.grant_type}")


@app.post("/oauth/introspect")
async def oauth_introspect(
    token: str = "",
    authorization: Optional[str] = Header(None),
):
    """
    Token introspection endpoint (RFC 7662).
    
    Requires authentication (for authorized servers only).
    """
    # Validate the requesting server
    try:
        await app_state.oauth_validator.validate_token(authorization)
    except PermissionError as e:
        raise HTTPException(status_code=401, detail=str(e))
    
    response = app_state.oauth_validator.introspect_token(token)
    return response.model_dump()


@app.post("/oauth/revoke")
async def oauth_revoke(
    token: Optional[str] = None,
    token_type_hint: str = "refresh_token",
    authorization: Optional[str] = Header(None),
):
    """
    Token revocation endpoint (RFC 7009).
    """
    try:
        user = await app_state.oauth_validator.validate_token(authorization)
    except PermissionError as e:
        raise HTTPException(status_code=401, detail=str(e))
    
    if not token:
        raise HTTPException(status_code=400, detail="Missing token parameter")
    
    async with get_db() as session:
        token_store = RefreshTokenStore(session)
        
        if token_type_hint == "refresh_token":
            # Validate and revoke refresh token
            token_data = await token_store.validate_refresh_token(token)
            if token_data and token_data.user_id == user.id:
                await token_store.revoke_user_tokens(user.id, user.client_id)
        
        # Access tokens are short-lived and self-expiring
        # We just acknowledge the revocation request
    
    return {"status": "revoked"}


# -----------------------------------------------------------------------------
# Consent Endpoints
# -----------------------------------------------------------------------------


@app.get("/consent")
async def list_consents(
    authorization: Optional[str] = Header(None),
):
    """List user's granted consents."""
    try:
        user = await app_state.oauth_validator.validate_token(authorization)
    except PermissionError as e:
        raise HTTPException(status_code=401, detail=str(e))
    
    async with get_db() as session:
        consent_db = ConsentDB(session)
        consents = await consent_db.list_user_consents(user.id)
        return {"consents": consents}


@app.post("/consent/grant")
async def grant_consent(
    request: ConsentGrantRequest,
    authorization: Optional[str] = Header(None),
):
    """Grant consent to a client."""
    try:
        user = await app_state.oauth_validator.validate_token(authorization)
    except PermissionError as e:
        raise HTTPException(status_code=401, detail=str(e))
    
    config = get_proxy_config()
    expires_days = request.expires_days or config.consent_default_expiry_days
    
    async with get_db() as session:
        consent_db = ConsentDB(session)
        await consent_db.grant_consent(
            user_id=user.id,
            client_id=request.client_id,
            scopes=request.scopes,
            expires_days=expires_days,
        )
    
    logger.info(
        "consent_granted",
        user_id=user.id,
        client_id=request.client_id,
        scopes=request.scopes,
    )
    
    return {"status": "granted", "client_id": request.client_id, "scopes": request.scopes}


@app.post("/consent/revoke")
async def revoke_consent(
    request: ConsentRevokeRequest,
    authorization: Optional[str] = Header(None),
):
    """Revoke consent from a client."""
    try:
        user = await app_state.oauth_validator.validate_token(authorization)
    except PermissionError as e:
        raise HTTPException(status_code=401, detail=str(e))
    
    async with get_db() as session:
        consent_db = ConsentDB(session)
        revoked = await consent_db.revoke_consent(user.id, request.client_id)
    
    if revoked:
        logger.info("consent_revoked", user_id=user.id, client_id=request.client_id)
        return {"status": "revoked", "client_id": request.client_id}
    else:
        return {"status": "not_found", "client_id": request.client_id}


@app.delete("/consent/{client_id}")
async def delete_client_consent(
    client_id: str,
    authorization: Optional[str] = Header(None),
):
    """Delete all consents for a client."""
    try:
        user = await app_state.oauth_validator.validate_token(authorization)
    except PermissionError as e:
        raise HTTPException(status_code=401, detail=str(e))
    
    async with get_db() as session:
        consent_db = ConsentDB(session)
        await consent_db.revoke_consent(user.id, client_id)
    
    return {"status": "deleted", "client_id": client_id}


# -----------------------------------------------------------------------------
# Audit Log Endpoints
# -----------------------------------------------------------------------------


@app.get("/audit/logs")
async def query_audit_logs(
    user_id: Optional[str] = None,
    client_id: Optional[str] = None,
    tool_name: Optional[str] = None,
    status: Optional[str] = None,
    anomaly_detected: Optional[bool] = None,
    page: int = 1,
    page_size: int = 50,
    authorization: Optional[str] = Header(None),
):
    """Query audit logs with filters."""
    try:
        user = await app_state.oauth_validator.validate_token(authorization)
    except PermissionError as e:
        raise HTTPException(status_code=401, detail=str(e))
    
    # Users can only query their own logs (unless admin)
    if user_id and user_id != user.id:
        # Check if user has admin scope
        if "admin" not in user.scopes:
            raise HTTPException(status_code=403, detail="Cannot query other users' logs")
    
    filters = AuditQueryFilters(
        user_id=user_id or user.id,  # Default to own logs
        client_id=client_id,
        tool_name=tool_name,
        status=status,
        anomaly_detected=anomaly_detected,
    )
    
    pagination = Pagination(page=page, page_size=page_size)
    
    async with get_db() as session:
        audit_db = AuditLogDB(session)
        result = await audit_db.query_logs(filters, pagination)
    
    return result.model_dump()


@app.get("/audit/summary")
async def get_activity_summary(
    days: int = 7,
    authorization: Optional[str] = Header(None),
):
    """Get user activity summary."""
    try:
        user = await app_state.oauth_validator.validate_token(authorization)
    except PermissionError as e:
        raise HTTPException(status_code=401, detail=str(e))
    
    async with get_db() as session:
        audit_db = AuditLogDB(session)
        summary = await audit_db.get_user_activity_summary(user.id, days)
    
    return summary.model_dump()


@app.get("/audit/export")
async def export_audit_logs(
    start: datetime,
    end: datetime,
    format: str = "json",
    authorization: Optional[str] = Header(None),
):
    """Export audit logs (admin only)."""
    try:
        user = await app_state.oauth_validator.validate_token(authorization)
    except PermissionError as e:
        raise HTTPException(status_code=401, detail=str(e))
    
    if "admin" not in user.scopes:
        raise HTTPException(status_code=403, detail="Admin access required")
    
    async with get_db() as session:
        audit_db = AuditLogDB(session)
        data = await audit_db.export_logs(start, end, format)
    
    media_type = "application/json" if format == "json" else "text/csv"
    filename = f"audit_logs_{start.strftime('%Y%m%d')}_{end.strftime('%Y%m%d')}.{format}"
    
    return Response(
        content=data,
        media_type=media_type,
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


# -----------------------------------------------------------------------------
# Tool Call Endpoint
# -----------------------------------------------------------------------------


@app.post("/tool_call", response_model=ToolResponse)
async def tool_call(
    request: ToolRequest,
    http_request: Request,
    response: Response,
    authorization: Optional[str] = Header(None),
    config: ProxyConfig = Depends(get_proxy_config),
):
    """
    Main tool call endpoint.

    Flow:
    1. Authenticate (JWT validation)
    2. Check confused deputy (consent)
    3. Check intent (LLM-as-Judge)
    4. Check behavioral anomalies (ML)
    5. Rate limit
    6. Audit log
    7. Forward to MCP server
    """
    start_time = time.perf_counter()
    client_ip = get_client_ip(http_request)
    request_id = generate_request_id()
    user_id: Optional[str] = None
    intent_result = None
    anomaly_result = None

    logger.info("tool_call_received", tool=request.tool_name, ip=client_ip, request_id=request_id)
    metrics.requests_total.labels(tool=request.tool_name).inc()

    try:
        # -------------------------
        # Step 1: Authenticate
        # -------------------------
        try:
            user = await app_state.oauth_validator.validate_token(authorization)
            user_id = user.id
            AUTH_ATTEMPTS.labels(status="success").inc()
            logger.info("auth_success", user_id=user_id)
        except PermissionError as e:
            AUTH_ATTEMPTS.labels(status="failed").inc()
            logger.warning("auth_failed", error=str(e), ip=client_ip)
            raise HTTPException(status_code=401, detail=str(e))

        # -------------------------
        # Step 2: Check Confused Deputy
        # -------------------------
        if config.consent_enabled and request.client_id:
            async with get_db() as session:
                consent_db = ConsentDB(session)
                has_consent = await consent_db.has_consent(user_id, request.client_id)
                if not has_consent:
                    logger.warning(
                        "consent_denied",
                        user_id=user_id,
                        client_id=request.client_id,
                    )
                    raise PermissionError(
                        f"Client '{request.client_id}' is not authorized for this user"
                    )

        # -------------------------
        # Step 3: Check Intent (LLM-as-Judge)
        # -------------------------
        if config.intent_check_enabled:
            intent_start = time.perf_counter()
            intent_result = await app_state.intent_checker.validate_intent(
                request.tool_name, request.params, request.user_stated_intent
            )
            intent_latency = time.perf_counter() - intent_start

            INTENT_CONFIDENCE.observe(intent_result.confidence)
            logger.info(
                "intent_check_complete",
                tool=request.tool_name,
                valid=intent_result.is_valid,
                confidence=intent_result.confidence,
                latency_ms=int(intent_latency * 1000),
            )

            if not intent_result.is_valid:
                metrics.intent_violations_total.labels(tool=request.tool_name).inc()
                raise PermissionError(f"Intent violation: {intent_result.reason}")

        # -------------------------
        # Step 4: Check Behavioral Anomalies
        # -------------------------
        async with get_db() as session:
            audit_log_db = AuditLogDB(session)
            user_history = await audit_log_db.get_recent_calls(user_id, limit=50)

        anomaly_request = AnomalyRequest(
            tool_name=request.tool_name,
            user_id=user_id,
            timestamp=datetime.utcnow(),
            ip=client_ip,
            params=request.params,
        )

        anomaly_result = app_state.anomaly_detector.check_anomaly(
            anomaly_request, user_history
        )

        ANOMALY_SCORE.observe(anomaly_result.score)
        logger.info(
            "anomaly_check_complete",
            tool=request.tool_name,
            is_anomalous=anomaly_result.is_anomalous,
            score=anomaly_result.score,
        )

        if anomaly_result.is_anomalous and anomaly_result.score > config.anomaly_threshold:
            metrics.anomalies_detected_total.labels(user=user_id).inc()
            raise PermissionError(f"Anomaly detected: {anomaly_result.reason}")

        # -------------------------
        # Step 5: Rate Limit
        # -------------------------
        if config.rate_limit_enabled:
            async with get_db() as session:
                rate_result = await rate_limiter.check_rate_limit(
                    user_id=user_id,
                    tool_name=request.tool_name,
                    session=session,
                )
            
            # Add rate limit headers
            for key, value in rate_result.get_headers().items():
                response.headers[key] = value
            
            if not rate_result.allowed:
                raise PermissionError("Rate limit exceeded")

        # -------------------------
        # Step 6: Forward to MCP Server
        # -------------------------
        mcp_response = await mcp_client.call_tool(
            tool_name=request.tool_name,
            params=request.params,
            user_id=user_id,
            client_id=request.client_id,
        )

        duration_ms = int((time.perf_counter() - start_time) * 1000)

        # -------------------------
        # Step 7: Audit Log
        # -------------------------
        if config.audit_enabled:
            async with get_db() as session:
                audit_log_db = AuditLogDB(session)
                await audit_log_db.record_request(
                    user_id=user_id,
                    tool_name=request.tool_name,
                    tool_input=json.dumps(request.params),
                    tool_output=json.dumps(mcp_response.result) if mcp_response.result else None,
                    status="success" if mcp_response.success else "error",
                    client_id=request.client_id,
                    duration_ms=duration_ms,
                    ip_address=client_ip,
                    request_id=request_id,
                    intent_check_passed=intent_result.is_valid if intent_result else None,
                    intent_confidence=intent_result.confidence if intent_result else None,
                    anomaly_score=anomaly_result.score if anomaly_result else None,
                    anomaly_detected=anomaly_result.is_anomalous if anomaly_result else None,
                    sanitize=config.audit_sanitize_outputs,
                )

        metrics.request_latency.labels(tool=request.tool_name).observe(
            time.perf_counter() - start_time
        )

        return ToolResponse(
            success=mcp_response.success,
            result=mcp_response.result,
            error=mcp_response.error,
            duration_ms=duration_ms,
        )

    except PermissionError as e:
        logger.warning("permission_denied", tool=request.tool_name, reason=str(e))
        metrics.requests_denied.labels(tool=request.tool_name).inc()

        # Record denied request
        if user_id and config.audit_enabled:
            async with get_db() as session:
                audit_log_db = AuditLogDB(session)
                await audit_log_db.record_request(
                    user_id=user_id,
                    tool_name=request.tool_name,
                    tool_input=json.dumps(request.params),
                    status="denied",
                    deny_reason=str(e),
                    client_id=request.client_id,
                    ip_address=client_ip,
                    request_id=request_id,
                    intent_check_passed=intent_result.is_valid if intent_result else None,
                    intent_confidence=intent_result.confidence if intent_result else None,
                    anomaly_score=anomaly_result.score if anomaly_result else None,
                    anomaly_detected=anomaly_result.is_anomalous if anomaly_result else None,
                    sanitize=config.audit_sanitize_outputs,
                )

        raise HTTPException(status_code=403, detail=str(e))

    except HTTPException:
        raise

    except Exception as e:
        logger.error("tool_call_error", tool=request.tool_name, error=str(e))
        metrics.requests_failed.labels(tool=request.tool_name).inc()
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/tools")
async def list_tools(authorization: Optional[str] = Header(None)):
    """List available MCP tools."""
    try:
        await app_state.oauth_validator.validate_token(authorization)
    except PermissionError as e:
        raise HTTPException(status_code=401, detail=str(e))

    return await mcp_client.list_tools()


# -----------------------------------------------------------------------------
# Main Entry Point
# -----------------------------------------------------------------------------


def main():
    """Main entry point for the proxy server."""
    import uvicorn

    config = get_proxy_config()
    uvicorn.run(
        "proxy.auth_gateway:app",
        host=config.host,
        port=config.port,
        reload=config.debug,
    )


if __name__ == "__main__":
    main()
