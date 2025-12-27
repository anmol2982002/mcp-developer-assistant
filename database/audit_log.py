"""
Audit Log Database Operations

Enhanced CRUD operations for audit logs with sanitization,
query API, and activity summaries.
"""

import json
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from pydantic import BaseModel
from sqlalchemy import and_, func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from database.sanitizer import sanitize_output, hash_pii
from database.schema import AuditLog
from observability.logging_config import get_logger

logger = get_logger(__name__)


# -----------------------------------------------------------------------------
# Models
# -----------------------------------------------------------------------------


class AuditQueryFilters(BaseModel):
    """Filters for audit log queries."""
    
    user_id: Optional[str] = None
    client_id: Optional[str] = None
    tool_name: Optional[str] = None
    status: Optional[str] = None  # success, denied, error
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    request_id: Optional[str] = None
    anomaly_detected: Optional[bool] = None
    intent_check_passed: Optional[bool] = None
    min_anomaly_score: Optional[float] = None
    ip_address: Optional[str] = None


class Pagination(BaseModel):
    """Pagination settings."""
    
    page: int = 1
    page_size: int = 50
    max_page_size: int = 100
    
    @property
    def offset(self) -> int:
        return (self.page - 1) * min(self.page_size, self.max_page_size)
    
    @property
    def limit(self) -> int:
        return min(self.page_size, self.max_page_size)


class PaginatedResult(BaseModel):
    """Paginated query result."""
    
    items: List[Dict[str, Any]]
    total: int
    page: int
    page_size: int
    total_pages: int
    has_next: bool
    has_prev: bool


class ActivitySummary(BaseModel):
    """User activity summary."""
    
    user_id: str
    period_days: int
    total_requests: int
    successful_requests: int
    denied_requests: int
    error_requests: int
    unique_tools: int
    top_tools: List[Dict[str, int]]
    anomalies_detected: int
    intent_violations: int
    avg_response_time_ms: Optional[float]
    unique_ips: int


# -----------------------------------------------------------------------------
# Audit Log Database
# -----------------------------------------------------------------------------


class AuditLogDB:
    """
    Enhanced audit log database operations.
    
    Features:
    - Sanitized recording (PII hashing, secret removal)
    - Advanced query API with filters and pagination
    - User activity summaries
    - Export functionality
    """

    def __init__(self, session: AsyncSession):
        self.session = session

    async def record_request(
        self,
        user_id: str,
        tool_name: str,
        tool_input: Optional[str] = None,
        tool_output: Optional[str] = None,
        status: str = "success",
        client_id: Optional[str] = None,
        duration_ms: Optional[int] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        request_id: Optional[str] = None,
        session_id: Optional[str] = None,
        intent_check_passed: Optional[bool] = None,
        intent_confidence: Optional[float] = None,
        anomaly_score: Optional[float] = None,
        anomaly_detected: Optional[bool] = None,
        deny_reason: Optional[str] = None,
        sanitize: bool = True,
    ) -> AuditLog:
        """
        Record a tool request with optional sanitization.
        
        Args:
            user_id: User identifier
            tool_name: Name of the tool called
            tool_input: Tool input (JSON string)
            tool_output: Tool output (will be hashed and sanitized)
            status: Request status (success, denied, error)
            client_id: Client application ID
            duration_ms: Request duration in milliseconds
            ip_address: Client IP address
            user_agent: Client user agent
            request_id: Unique request ID for correlation
            session_id: Session ID for tracking
            intent_check_passed: Whether intent check passed
            intent_confidence: Intent check confidence score
            anomaly_score: Anomaly detection score
            anomaly_detected: Whether anomaly was detected
            deny_reason: Reason for denial (if denied)
            sanitize: Whether to sanitize inputs/outputs
            
        Returns:
            Created audit log record
        """
        # Generate request ID if not provided
        if not request_id:
            request_id = str(uuid.uuid4())
        
        # Sanitize inputs if enabled
        sanitized_input = tool_input
        sanitized_output = None
        output_hash = None
        
        if sanitize and tool_input:
            try:
                input_data = json.loads(tool_input)
                sanitized_input = json.dumps(sanitize_output(input_data))
            except (json.JSONDecodeError, Exception):
                sanitized_input = sanitize_output(tool_input)
        
        if tool_output:
            # Hash the output for integrity
            import hashlib
            output_hash = hashlib.sha256(tool_output.encode()).hexdigest()
            
            # Sanitize output for storage
            if sanitize:
                try:
                    output_data = json.loads(tool_output)
                    sanitized_output = json.dumps(sanitize_output(output_data))
                except (json.JSONDecodeError, Exception):
                    sanitized_output = sanitize_output(tool_output)
            else:
                sanitized_output = tool_output
        
        # Hash IP address for privacy
        hashed_ip = hash_pii(ip_address) if ip_address and sanitize else ip_address
        
        log = AuditLog(
            user_id=user_id,
            client_id=client_id,
            tool_name=tool_name,
            tool_input=sanitized_input[:1000] if sanitized_input else None,  # Truncate
            tool_output_hash=output_hash,
            tool_output_sanitized=sanitized_output[:2000] if sanitized_output else None,
            status=status,
            deny_reason=deny_reason,
            duration_ms=duration_ms,
            ip_address=hashed_ip,
            user_agent=user_agent[:500] if user_agent else None,
            request_id=request_id,
            session_id=session_id,
            intent_check_passed=intent_check_passed,
            intent_confidence=intent_confidence,
            anomaly_score=anomaly_score,
            anomaly_detected=anomaly_detected,
        )

        self.session.add(log)
        await self.session.commit()

        logger.debug(
            "audit_recorded",
            user_id=user_id,
            tool=tool_name,
            request_id=request_id,
        )
        
        return log

    async def query_logs(
        self,
        filters: AuditQueryFilters,
        pagination: Optional[Pagination] = None,
    ) -> PaginatedResult:
        """
        Query audit logs with filters and pagination.
        
        Args:
            filters: Query filters
            pagination: Pagination settings
            
        Returns:
            Paginated result with matching logs
        """
        pagination = pagination or Pagination()
        
        # Build query conditions
        conditions = []
        
        if filters.user_id:
            conditions.append(AuditLog.user_id == filters.user_id)
        
        if filters.client_id:
            conditions.append(AuditLog.client_id == filters.client_id)
        
        if filters.tool_name:
            conditions.append(AuditLog.tool_name == filters.tool_name)
        
        if filters.status:
            conditions.append(AuditLog.status == filters.status)
        
        if filters.start_time:
            conditions.append(AuditLog.timestamp >= filters.start_time)
        
        if filters.end_time:
            conditions.append(AuditLog.timestamp <= filters.end_time)
        
        if filters.request_id:
            conditions.append(AuditLog.request_id == filters.request_id)
        
        if filters.anomaly_detected is not None:
            conditions.append(AuditLog.anomaly_detected == filters.anomaly_detected)
        
        if filters.intent_check_passed is not None:
            conditions.append(AuditLog.intent_check_passed == filters.intent_check_passed)
        
        if filters.min_anomaly_score is not None:
            conditions.append(AuditLog.anomaly_score >= filters.min_anomaly_score)
        
        if filters.ip_address:
            # Match hashed IP
            hashed_ip = hash_pii(filters.ip_address)
            conditions.append(AuditLog.ip_address == hashed_ip)
        
        # Count total
        count_query = select(func.count(AuditLog.id))
        if conditions:
            count_query = count_query.where(and_(*conditions))
        
        count_result = await self.session.execute(count_query)
        total = count_result.scalar()
        
        # Get paginated results
        query = select(AuditLog)
        if conditions:
            query = query.where(and_(*conditions))
        
        query = (
            query
            .order_by(AuditLog.timestamp.desc())
            .offset(pagination.offset)
            .limit(pagination.limit)
        )
        
        result = await self.session.execute(query)
        logs = result.scalars().all()
        
        # Convert to dicts
        items = [
            {
                "id": log.id,
                "user_id": log.user_id,
                "client_id": log.client_id,
                "tool_name": log.tool_name,
                "status": log.status,
                "timestamp": log.timestamp.isoformat() if log.timestamp else None,
                "duration_ms": log.duration_ms,
                "request_id": log.request_id,
                "anomaly_score": log.anomaly_score,
                "anomaly_detected": log.anomaly_detected,
                "intent_check_passed": log.intent_check_passed,
                "intent_confidence": log.intent_confidence,
            }
            for log in logs
        ]
        
        total_pages = (total + pagination.limit - 1) // pagination.limit
        
        return PaginatedResult(
            items=items,
            total=total,
            page=pagination.page,
            page_size=pagination.limit,
            total_pages=total_pages,
            has_next=pagination.page < total_pages,
            has_prev=pagination.page > 1,
        )

    async def get_recent_calls(
        self,
        user_id: str,
        limit: int = 50,
        hours: int = 24,
    ) -> List[dict]:
        """Get recent calls for a user."""
        cutoff = datetime.utcnow() - timedelta(hours=hours)

        query = (
            select(AuditLog)
            .where(AuditLog.user_id == user_id)
            .where(AuditLog.timestamp > cutoff)
            .order_by(AuditLog.timestamp.desc())
            .limit(limit)
        )

        result = await self.session.execute(query)
        logs = result.scalars().all()

        return [
            {
                "tool": log.tool_name,
                "timestamp": log.timestamp,
                "ip": log.ip_address,
                "status": log.status,
            }
            for log in logs
        ]

    async def get_user_activity_summary(
        self,
        user_id: str,
        days: int = 7,
    ) -> ActivitySummary:
        """
        Get activity summary for a user.
        
        Args:
            user_id: User identifier
            days: Number of days to summarize
            
        Returns:
            Activity summary with statistics
        """
        cutoff = datetime.utcnow() - timedelta(days=days)
        
        # Base query
        base_condition = and_(
            AuditLog.user_id == user_id,
            AuditLog.timestamp > cutoff,
        )
        
        # Total requests
        total_query = select(func.count(AuditLog.id)).where(base_condition)
        total_result = await self.session.execute(total_query)
        total_requests = total_result.scalar()
        
        # Status breakdown
        status_query = (
            select(AuditLog.status, func.count(AuditLog.id))
            .where(base_condition)
            .group_by(AuditLog.status)
        )
        status_result = await self.session.execute(status_query)
        status_counts = {row[0]: row[1] for row in status_result.fetchall()}
        
        # Unique tools
        tools_query = select(func.count(func.distinct(AuditLog.tool_name))).where(base_condition)
        tools_result = await self.session.execute(tools_query)
        unique_tools = tools_result.scalar()
        
        # Top tools
        top_tools_query = (
            select(AuditLog.tool_name, func.count(AuditLog.id).label("count"))
            .where(base_condition)
            .group_by(AuditLog.tool_name)
            .order_by(func.count(AuditLog.id).desc())
            .limit(5)
        )
        top_tools_result = await self.session.execute(top_tools_query)
        top_tools = [{"tool": row[0], "count": row[1]} for row in top_tools_result.fetchall()]
        
        # Anomalies
        anomaly_query = (
            select(func.count(AuditLog.id))
            .where(and_(base_condition, AuditLog.anomaly_detected == True))
        )
        anomaly_result = await self.session.execute(anomaly_query)
        anomalies = anomaly_result.scalar()
        
        # Intent violations
        intent_query = (
            select(func.count(AuditLog.id))
            .where(and_(base_condition, AuditLog.intent_check_passed == False))
        )
        intent_result = await self.session.execute(intent_query)
        intent_violations = intent_result.scalar()
        
        # Average response time
        avg_time_query = select(func.avg(AuditLog.duration_ms)).where(base_condition)
        avg_time_result = await self.session.execute(avg_time_query)
        avg_time = avg_time_result.scalar()
        
        # Unique IPs
        ips_query = select(func.count(func.distinct(AuditLog.ip_address))).where(base_condition)
        ips_result = await self.session.execute(ips_query)
        unique_ips = ips_result.scalar()
        
        return ActivitySummary(
            user_id=user_id,
            period_days=days,
            total_requests=total_requests or 0,
            successful_requests=status_counts.get("success", 0),
            denied_requests=status_counts.get("denied", 0),
            error_requests=status_counts.get("error", 0),
            unique_tools=unique_tools or 0,
            top_tools=top_tools,
            anomalies_detected=anomalies or 0,
            intent_violations=intent_violations or 0,
            avg_response_time_ms=round(avg_time, 2) if avg_time else None,
            unique_ips=unique_ips or 0,
        )

    async def get_logs(
        self,
        hours: int = 24,
        tool_name: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> List[AuditLog]:
        """Get logs with filters (legacy API)."""
        cutoff = datetime.utcnow() - timedelta(hours=hours)

        query = select(AuditLog).where(AuditLog.timestamp > cutoff)

        if tool_name:
            query = query.where(AuditLog.tool_name == tool_name)
        if user_id:
            query = query.where(AuditLog.user_id == user_id)

        query = query.order_by(AuditLog.timestamp.desc())

        result = await self.session.execute(query)
        return result.scalars().all()

    async def get_anomalies(self, hours: int = 24) -> List[AuditLog]:
        """Get detected anomalies."""
        cutoff = datetime.utcnow() - timedelta(hours=hours)

        query = (
            select(AuditLog)
            .where(AuditLog.timestamp > cutoff)
            .where(AuditLog.anomaly_detected == True)
            .order_by(AuditLog.timestamp.desc())
        )

        result = await self.session.execute(query)
        return result.scalars().all()

    async def export_logs(
        self,
        start: datetime,
        end: datetime,
        format: str = "json",
    ) -> bytes:
        """
        Export logs for a time period.
        
        Args:
            start: Start datetime
            end: End datetime
            format: Export format (json, csv)
            
        Returns:
            Exported data as bytes
        """
        query = (
            select(AuditLog)
            .where(and_(
                AuditLog.timestamp >= start,
                AuditLog.timestamp <= end,
            ))
            .order_by(AuditLog.timestamp)
        )
        
        result = await self.session.execute(query)
        logs = result.scalars().all()
        
        if format == "json":
            import json
            data = [
                {
                    "id": log.id,
                    "user_id": log.user_id,
                    "client_id": log.client_id,
                    "tool_name": log.tool_name,
                    "status": log.status,
                    "timestamp": log.timestamp.isoformat() if log.timestamp else None,
                    "duration_ms": log.duration_ms,
                    "request_id": log.request_id,
                    "anomaly_score": log.anomaly_score,
                    "intent_confidence": log.intent_confidence,
                }
                for log in logs
            ]
            return json.dumps(data, indent=2).encode()
        
        elif format == "csv":
            import csv
            import io
            
            output = io.StringIO()
            writer = csv.writer(output)
            
            # Header
            writer.writerow([
                "id", "user_id", "client_id", "tool_name", "status",
                "timestamp", "duration_ms", "request_id",
                "anomaly_score", "intent_confidence",
            ])
            
            # Rows
            for log in logs:
                writer.writerow([
                    log.id,
                    log.user_id,
                    log.client_id,
                    log.tool_name,
                    log.status,
                    log.timestamp.isoformat() if log.timestamp else "",
                    log.duration_ms or "",
                    log.request_id or "",
                    log.anomaly_score or "",
                    log.intent_confidence or "",
                ])
            
            return output.getvalue().encode()
        
        else:
            raise ValueError(f"Unsupported format: {format}")
