"""
Audit logging service implementation for security compliance and tamper-resistant logging.

This service provides comprehensive audit logging capabilities with tamper resistance,
compliance reporting, and security event tracking following hexagonal architecture principles.
"""

import json
import hashlib
import csv
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import uuid4
from io import StringIO

import structlog
from sqlalchemy import and_, func, or_, text
from sqlalchemy.orm import selectinload

from app.domain.interfaces import IAuditService
from app.infrastructure.persistence.models.audit_table import (
    AuditLogTable,
    AuditEventType,
    AuditRiskLevel,
    AuditLogCreate,
    AuditLogResponse,
    AuditLogQuery,
    AuditLogStats,
)
from app.services.adapters.postgres_adapter import PostgresAdapter
from app.database.error_handling import handle_database_errors, log_database_operation

logger = structlog.get_logger(__name__)


class AuditServiceError(Exception):
    """Raised when audit service operations fail."""
    
    def __init__(self, message: str, error_code: str = "AUDIT_ERROR"):
        super().__init__(message)
        self.message = message
        self.error_code = error_code


class AuditService(IAuditService):
    """
    Comprehensive audit logging service with tamper resistance and compliance features.
    
    This service implements audit logging with:
    - Tamper-resistant log integrity using SHA-256 hashing
    - Sequence numbering for gap detection
    - Comprehensive event categorization and risk assessment
    - Performance-optimized queries with proper indexing
    - Compliance reporting and export capabilities
    """

    def __init__(self, database_adapter: PostgresAdapter):
        """
        Initialize audit service with database adapter.
        
        Args:
            database_adapter: Database adapter for audit log persistence
        """
        self.db_adapter = database_adapter
        self._sequence_counters = {}  # In-memory sequence counters per tenant
        
    async def check_health(self) -> Dict[str, Any]:
        """Check audit service health status."""
        try:
            # Test database connectivity
            row = await self.db_adapter.fetch_one("SELECT 1")

            # Check audit logs table exists
            row = await self.db_adapter.fetch_one(
                """
                SELECT EXISTS (
                    SELECT FROM information_schema.tables
                    WHERE table_schema = 'public'
                    AND table_name = 'audit_logs'
                )
                """
            )
            result = list(row.values())[0] if row else None
            
            table_exists = bool(result)
            
            return {
                "status": "healthy" if table_exists else "degraded",
                "database_connected": True,
                "audit_table_exists": table_exists,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        except Exception as e:
            logger.error("Audit service health check failed", error=str(e))
            return {
                "status": "unhealthy",
                "database_connected": False,
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

    @handle_database_errors(context={"operation": "audit_log_creation"})
    @log_database_operation("log_audit_event")
    async def log_event(
        self,
        event_type: str,
        tenant_id: str,
        action: str,
        resource_type: str,
        *,
        user_id: Optional[str] = None,
        user_email: Optional[str] = None,
        session_id: Optional[str] = None,
        api_key_id: Optional[str] = None,
        resource_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        risk_level: str = "low",
        suspicious: bool = False,
        correlation_id: Optional[str] = None,
        batch_id: Optional[str] = None,
        error_code: Optional[str] = None,
        error_message: Optional[str] = None,
    ) -> str:
        """Log an audit event with comprehensive details."""
        try:
            # Generate audit log entry ID
            log_id = str(uuid4())
            current_time = datetime.utcnow()
            
            # Get next sequence number for this tenant
            sequence_number = await self._get_next_sequence_number(tenant_id)
            
            # Prepare audit log data
            audit_data = {
                "id": log_id,
                "tenant_id": tenant_id,
                "event_type": event_type,
                "user_id": user_id,
                "user_email": user_email,
                "session_id": session_id,
                "api_key_id": api_key_id,
                "resource_type": resource_type,
                "resource_id": resource_id,
                "action": action,
                "details": details or {},
                "ip_address": ip_address,
                "user_agent": user_agent,
                "risk_level": risk_level,
                "suspicious": suspicious,
                "event_timestamp": current_time,
                "logged_at": current_time,
                "sequence_number": sequence_number,
                "correlation_id": correlation_id,
                "batch_id": batch_id,
                "error_code": error_code,
                "error_message": error_message,
            }
            
            # Generate tamper-resistant hash
            log_hash = self._generate_log_hash(audit_data)
            audit_data["log_hash"] = log_hash
            
            # Insert audit log entry
            await self.db_adapter.execute(
                """
                INSERT INTO audit_logs (
                    id, created_at, updated_at, tenant_id, event_type, user_id, user_email, session_id, api_key_id,
                    resource_type, resource_id, action, details, ip_address, user_agent,
                    risk_level, suspicious, event_timestamp, logged_at, log_hash,
                    sequence_number, correlation_id, batch_id, error_code, error_message
                ) VALUES (
                    $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15,
                    $16, $17, $18, $19, $20, $21, $22, $23, $24, $25
                )
                """,
                log_id, current_time, current_time, tenant_id, event_type, user_id, user_email, session_id, api_key_id,
                resource_type, resource_id, action, json.dumps(details or {}), ip_address,
                user_agent, risk_level, suspicious, current_time, current_time, log_hash,
                sequence_number, correlation_id, batch_id, error_code, error_message
            )
            
            logger.info(
                "Audit event logged successfully",
                log_id=log_id,
                tenant_id=tenant_id,
                event_type=event_type,
                action=action,
                resource_type=resource_type,
                risk_level=risk_level,
                suspicious=suspicious,
                sequence_number=sequence_number,
            )
            
            return log_id
            
        except Exception as e:
            logger.error(
                "Failed to log audit event",
                error=str(e),
                tenant_id=tenant_id,
                event_type=event_type,
                action=action,
                resource_type=resource_type,
            )
            raise AuditServiceError(f"Failed to log audit event: {str(e)}")

    async def log_authentication_event(
        self,
        event_type: str,
        tenant_id: str,
        user_id: Optional[str] = None,
        user_email: Optional[str] = None,
        session_id: Optional[str] = None,
        ip_address: str = "unknown",
        user_agent: str = "unknown",
        success: bool = True,
        failure_reason: Optional[str] = None,
        additional_details: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Log authentication-specific events with standardized details."""
        details = {
            "success": success,
            "failure_reason": failure_reason,
            **(additional_details or {}),
        }
        
        risk_level = "low" if success else "medium"
        suspicious = not success and failure_reason in [
            "multiple_failed_attempts", "suspicious_location", "unusual_timing"
        ]
        
        return await self.log_event(
            event_type=event_type,
            tenant_id=tenant_id,
            action="authenticate" if success else "authentication_failed",
            resource_type="user_session",
            user_id=user_id,
            user_email=user_email,
            session_id=session_id,
            details=details,
            ip_address=ip_address,
            user_agent=user_agent,
            risk_level=risk_level,
            suspicious=suspicious,
            error_code=None if success else "AUTH_FAILED",
            error_message=failure_reason if not success else None,
        )

    async def log_file_upload_event(
        self,
        event_type: str,
        tenant_id: str,
        user_id: str,
        filename: str,
        file_size: int,
        upload_id: str,
        *,
        session_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        validation_errors: Optional[List[str]] = None,
        security_warnings: Optional[List[str]] = None,
        processing_duration_ms: Optional[int] = None,
        batch_id: Optional[str] = None,
        error_message: Optional[str] = None,
        additional_data: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Log file upload and processing events with file-specific details."""
        details = {
            "filename": filename,
            "file_size": file_size,
            "file_size_mb": round(file_size / (1024 * 1024), 2),
            "validation_errors": validation_errors or [],
            "security_warnings": security_warnings or [],
            "processing_duration_ms": processing_duration_ms,
            **(additional_data or {}),
        }
        
        # Determine risk level based on validation and security issues
        risk_level = "low"
        suspicious = False
        
        if security_warnings:
            risk_level = "high"
            suspicious = True
        elif validation_errors:
            risk_level = "medium"
        elif event_type in [AuditEventType.FILE_UPLOAD_FAILED, AuditEventType.FILE_VALIDATION_FAILED]:
            risk_level = "medium"
        
        return await self.log_event(
            event_type=event_type,
            tenant_id=tenant_id,
            action="file_upload",
            resource_type="file",
            user_id=user_id,
            session_id=session_id,
            resource_id=upload_id,
            details=details,
            ip_address=ip_address,
            user_agent=user_agent,
            risk_level=risk_level,
            suspicious=suspicious,
            batch_id=batch_id,
            error_code="FILE_UPLOAD_ERROR" if error_message else None,
            error_message=error_message,
        )

    async def log_security_event(
        self,
        event_type: str,
        tenant_id: str,
        threat_type: str,
        severity: str,
        *,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        resource_id: Optional[str] = None,
        ip_address: str = "unknown",
        user_agent: str = "unknown",
        threat_details: Optional[Dict[str, Any]] = None,
        mitigation_actions: Optional[List[str]] = None,
    ) -> str:
        """Log security-related events with threat analysis details."""
        details = {
            "threat_type": threat_type,
            "severity": severity,
            "threat_details": threat_details or {},
            "mitigation_actions": mitigation_actions or [],
        }
        
        # Map severity to risk level
        severity_risk_mapping = {
            "low": "low",
            "medium": "medium", 
            "high": "high",
            "critical": "critical",
        }
        
        return await self.log_event(
            event_type=event_type,
            tenant_id=tenant_id,
            action="security_threat_detected",
            resource_type="security",
            user_id=user_id,
            session_id=session_id,
            resource_id=resource_id,
            details=details,
            ip_address=ip_address,
            user_agent=user_agent,
            risk_level=severity_risk_mapping.get(severity, "medium"),
            suspicious=True,  # All security events are inherently suspicious
        )

    @handle_database_errors(context={"operation": "audit_log_query"})
    async def query_audit_logs(
        self,
        tenant_id: str,
        *,
        user_id: Optional[str] = None,
        event_types: Optional[List[str]] = None,
        resource_type: Optional[str] = None,
        resource_id: Optional[str] = None,
        risk_level: Optional[str] = None,
        suspicious_only: bool = False,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        correlation_id: Optional[str] = None,
        batch_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        page: int = 1,
        size: int = 50,
    ) -> Dict[str, Any]:
        """Query audit logs with filtering and pagination."""
        try:
            # Build WHERE conditions
            conditions = ["tenant_id = $1"]
            params = [tenant_id]
            param_count = 1
            
            if user_id:
                param_count += 1
                conditions.append(f"user_id = ${param_count}")
                params.append(user_id)
            
            if event_types:
                param_count += 1
                conditions.append(f"event_type = ANY(${param_count})")
                params.append(event_types)
            
            if resource_type:
                param_count += 1
                conditions.append(f"resource_type = ${param_count}")
                params.append(resource_type)
            
            if resource_id:
                param_count += 1
                conditions.append(f"resource_id = ${param_count}")
                params.append(resource_id)
            
            if risk_level:
                param_count += 1
                conditions.append(f"risk_level = ${param_count}")
                params.append(risk_level)
            
            if suspicious_only:
                param_count += 1
                conditions.append(f"suspicious = ${param_count}")
                params.append(True)
            
            if start_time:
                param_count += 1
                conditions.append(f"event_timestamp >= ${param_count}")
                params.append(start_time)
            
            if end_time:
                param_count += 1
                conditions.append(f"event_timestamp <= ${param_count}")
                params.append(end_time)
            
            if correlation_id:
                param_count += 1
                conditions.append(f"correlation_id = ${param_count}")
                params.append(correlation_id)
            
            if batch_id:
                param_count += 1
                conditions.append(f"batch_id = ${param_count}")
                params.append(batch_id)
            
            if ip_address:
                param_count += 1
                conditions.append(f"ip_address = ${param_count}")
                params.append(ip_address)
            
            where_clause = " AND ".join(conditions)
            
            # Calculate offset for pagination
            offset = (page - 1) * size
            
            # Get total count
            count_query = f"""
                SELECT COUNT(*) FROM audit_logs
                WHERE {where_clause}
            """
            row = await self.db_adapter.fetch_one(count_query, *params)
            total_count = list(row.values())[0] if row else 0

            # Get paginated results
            param_count += 1
            limit_param = f"${param_count}"
            param_count += 1
            offset_param = f"${param_count}"

            data_query = f"""
                SELECT id, tenant_id, event_type, user_id, user_email, session_id, api_key_id,
                       resource_type, resource_id, action, details, ip_address, user_agent,
                       risk_level, suspicious, event_timestamp, logged_at, correlation_id,
                       batch_id, error_code, error_message, sequence_number
                FROM audit_logs
                WHERE {where_clause}
                ORDER BY event_timestamp DESC, sequence_number DESC
                LIMIT {limit_param} OFFSET {offset_param}
            """

            rows = await self.db_adapter.fetch_all(
                data_query,
                *params, size, offset
            )
            
            # Convert rows to audit log responses
            audit_logs = []
            for row in rows:
                audit_log = AuditLogResponse(
                    id=str(row["id"]),
                    tenant_id=str(row["tenant_id"]),
                    event_type=row["event_type"],
                    user_id=str(row["user_id"]) if row["user_id"] else None,
                    user_email=row["user_email"],
                    resource_type=row["resource_type"],
                    resource_id=row["resource_id"],
                    action=row["action"],
                    details=row["details"] if isinstance(row["details"], dict) else json.loads(row["details"] or "{}"),
                    ip_address=str(row["ip_address"]),
                    risk_level=row["risk_level"],
                    suspicious=row["suspicious"],
                    event_timestamp=row["event_timestamp"].isoformat(),
                    correlation_id=row["correlation_id"],
                )
                audit_logs.append(audit_log.model_dump())
            
            # Calculate pagination metadata
            total_pages = (total_count + size - 1) // size
            
            return {
                "audit_logs": audit_logs,
                "pagination": {
                    "page": page,
                    "size": size,
                    "total": total_count,
                    "pages": total_pages,
                    "has_next": page < total_pages,
                    "has_prev": page > 1,
                },
            }
            
        except Exception as e:
            logger.error("Failed to query audit logs", error=str(e), tenant_id=tenant_id)
            raise AuditServiceError(f"Failed to query audit logs: {str(e)}")

    @handle_database_errors(context={"operation": "audit_statistics"})
    async def get_audit_statistics(
        self,
        tenant_id: str,
        *,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """Get audit log statistics for compliance reporting."""
        try:
            # Default time range to last 30 days if not specified
            if not end_time:
                end_time = datetime.now(timezone.utc)
            if not start_time:
                start_time = datetime.now(timezone.utc).replace(
                    day=1, hour=0, minute=0, second=0, microsecond=0
                )
            
            # Build base conditions
            base_conditions = "tenant_id = $1"
            params = [tenant_id]
            
            if start_time:
                base_conditions += " AND event_timestamp >= $2"
                params.append(start_time)
            if end_time:
                base_conditions += f" AND event_timestamp <= ${len(params) + 1}"
                params.append(end_time)
            
            # Get total events count
            row = await self.db_adapter.fetch_one(
                f"SELECT COUNT(*) FROM audit_logs WHERE {base_conditions}",
                *params
            )
            total_events = list(row.values())[0] if row else 0

            # Get events by type
            events_by_type = await self.db_adapter.fetch_all(
                f"""
                SELECT event_type, COUNT(*) as count
                FROM audit_logs
                WHERE {base_conditions}
                GROUP BY event_type
                ORDER BY count DESC
                """,
                *params
            )

            # Get events by risk level
            events_by_risk = await self.db_adapter.fetch_all(
                f"""
                SELECT risk_level, COUNT(*) as count
                FROM audit_logs
                WHERE {base_conditions}
                GROUP BY risk_level
                ORDER BY count DESC
                """,
                *params
            )

            # Get suspicious events count
            row = await self.db_adapter.fetch_one(
                f"SELECT COUNT(*) FROM audit_logs WHERE {base_conditions} AND suspicious = true",
                *params
            )
            suspicious_events = list(row.values())[0] if row else 0

            # Get recent events (last 24 hours)
            recent_time = datetime.now(timezone.utc).replace(
                hour=datetime.now(timezone.utc).hour - 24
            )
            row = await self.db_adapter.fetch_one(
                f"""
                SELECT COUNT(*) FROM audit_logs
                WHERE {base_conditions} AND event_timestamp >= ${len(params) + 1}
                """,
                *params, recent_time
            )
            recent_events = list(row.values())[0] if row else 0

            # Get unique users and IP addresses
            row = await self.db_adapter.fetch_one(
                f"""
                SELECT COUNT(DISTINCT user_id) FROM audit_logs
                WHERE {base_conditions} AND user_id IS NOT NULL
                """,
                *params
            )
            unique_users = list(row.values())[0] if row else 0

            row = await self.db_adapter.fetch_one(
                f"""
                SELECT COUNT(DISTINCT ip_address) FROM audit_logs
                WHERE {base_conditions}
                """,
                *params
            )
            unique_ips = list(row.values())[0] if row else 0
            
            return {
                "total_events": total_events or 0,
                "events_by_type": {row["event_type"]: row["count"] for row in events_by_type},
                "events_by_risk_level": {row["risk_level"]: row["count"] for row in events_by_risk},
                "suspicious_events": suspicious_events or 0,
                "recent_events": recent_events or 0,
                "unique_users": unique_users or 0,
                "unique_ip_addresses": unique_ips or 0,
                "time_range": {
                    "start_time": start_time.isoformat(),
                    "end_time": end_time.isoformat(),
                },
            }
            
        except Exception as e:
            logger.error("Failed to get audit statistics", error=str(e), tenant_id=tenant_id)
            raise AuditServiceError(f"Failed to get audit statistics: {str(e)}")

    async def verify_log_integrity(
        self,
        tenant_id: str,
        log_id: str,
    ) -> Dict[str, Any]:
        """Verify the integrity of a specific audit log entry."""
        try:
            # Get the audit log entry
            row = await self.db_adapter.fetch_one(
                """
                SELECT id, tenant_id, event_type, user_id, user_email, session_id, api_key_id,
                       resource_type, resource_id, action, details, ip_address, user_agent,
                       risk_level, suspicious, event_timestamp, logged_at, log_hash,
                       sequence_number, correlation_id, batch_id, error_code, error_message
                FROM audit_logs
                WHERE id = $1 AND tenant_id = $2
                """,
                log_id, tenant_id
            )
            
            if not row:
                return {
                    "verified": False,
                    "error": "Audit log entry not found",
                    "log_id": log_id,
                }
            
            # Reconstruct the audit data for hash verification
            audit_data = {
                "id": str(row["id"]),
                "tenant_id": str(row["tenant_id"]),
                "event_type": row["event_type"],
                "user_id": str(row["user_id"]) if row["user_id"] else None,
                "user_email": row["user_email"],
                "session_id": row["session_id"],
                "api_key_id": str(row["api_key_id"]) if row["api_key_id"] else None,
                "resource_type": row["resource_type"],
                "resource_id": row["resource_id"],
                "action": row["action"],
                "details": row["details"] if isinstance(row["details"], dict) else json.loads(row["details"] or "{}"),
                "ip_address": str(row["ip_address"]),
                "user_agent": row["user_agent"],
                "risk_level": row["risk_level"],
                "suspicious": row["suspicious"],
                "event_timestamp": row["event_timestamp"],
                "logged_at": row["logged_at"],
                "sequence_number": row["sequence_number"],
                "correlation_id": row["correlation_id"],
                "batch_id": row["batch_id"],
                "error_code": row["error_code"],
                "error_message": row["error_message"],
            }
            
            # Generate expected hash
            expected_hash = self._generate_log_hash(audit_data)
            stored_hash = row["log_hash"]
            
            # Verify integrity
            verified = expected_hash == stored_hash
            
            return {
                "verified": verified,
                "log_id": log_id,
                "stored_hash": stored_hash,
                "expected_hash": expected_hash,
                "event_timestamp": row["event_timestamp"].isoformat(),
                "sequence_number": row["sequence_number"],
            }
            
        except Exception as e:
            logger.error("Failed to verify log integrity", error=str(e), log_id=log_id)
            raise AuditServiceError(f"Failed to verify log integrity: {str(e)}")

    async def export_audit_logs(
        self,
        tenant_id: str,
        format: str = "json",
        *,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        event_types: Optional[List[str]] = None,
    ) -> bytes:
        """Export audit logs for compliance reporting."""
        try:
            # Build query conditions
            conditions = ["tenant_id = $1"]
            params = [tenant_id]
            param_count = 1
            
            if start_time:
                param_count += 1
                conditions.append(f"event_timestamp >= ${param_count}")
                params.append(start_time)
            
            if end_time:
                param_count += 1
                conditions.append(f"event_timestamp <= ${param_count}")
                params.append(end_time)
            
            if event_types:
                param_count += 1
                conditions.append(f"event_type = ANY(${param_count})")
                params.append(event_types)
            
            where_clause = " AND ".join(conditions)
            
            # Get audit logs
            query = f"""
                SELECT id, tenant_id, event_type, user_id, user_email, session_id, api_key_id,
                       resource_type, resource_id, action, details, ip_address, user_agent,
                       risk_level, suspicious, event_timestamp, logged_at, correlation_id,
                       batch_id, error_code, error_message, sequence_number
                FROM audit_logs
                WHERE {where_clause}
                ORDER BY event_timestamp ASC, sequence_number ASC
            """

            rows = await self.db_adapter.fetch_all(query, *params)
            
            # Export in requested format
            if format.lower() == "json":
                return self._export_as_json(rows)
            elif format.lower() == "csv":
                return self._export_as_csv(rows)
            elif format.lower() == "xml":
                return self._export_as_xml(rows)
            else:
                raise AuditServiceError(f"Unsupported export format: {format}")
                
        except Exception as e:
            logger.error("Failed to export audit logs", error=str(e), tenant_id=tenant_id)
            raise AuditServiceError(f"Failed to export audit logs: {str(e)}")

    async def _get_next_sequence_number(self, tenant_id: str) -> int:
        """Get the next sequence number for a tenant."""
        try:
            # Get the current max sequence number for this tenant
            row = await self.db_adapter.fetch_one(
                "SELECT COALESCE(MAX(sequence_number), 0) FROM audit_logs WHERE tenant_id = $1",
                tenant_id
            )
            result = list(row.values())[0] if row else 0

            return (result or 0) + 1

        except Exception as e:
            logger.warning("Failed to get sequence number, using timestamp", error=str(e))
            # Fallback to timestamp-based sequence
            return int(datetime.now(timezone.utc).timestamp() * 1000)

    def _generate_log_hash(self, audit_data: Dict[str, Any]) -> str:
        """Generate SHA-256 hash for tamper detection."""
        # Create a normalized string representation for hashing
        # Exclude the hash field itself and ensure consistent ordering
        hash_data = {
            k: v for k, v in audit_data.items() 
            if k not in ["log_hash"] and v is not None
        }
        
        # Convert datetime objects to ISO strings for consistent hashing
        for key, value in hash_data.items():
            if isinstance(value, datetime):
                hash_data[key] = value.isoformat()
        
        # Create deterministic string representation
        normalized_string = json.dumps(hash_data, sort_keys=True, separators=(',', ':'))
        
        # Generate SHA-256 hash
        return hashlib.sha256(normalized_string.encode('utf-8')).hexdigest()

    def _export_as_json(self, rows: List[Dict[str, Any]]) -> bytes:
        """Export audit logs as JSON."""
        export_data = []
        for row in rows:
            # Convert datetime objects to ISO strings
            row_data = dict(row)
            for key, value in row_data.items():
                if isinstance(value, datetime):
                    row_data[key] = value.isoformat()
            export_data.append(row_data)
        
        return json.dumps(export_data, indent=2).encode('utf-8')

    def _export_as_csv(self, rows: List[Dict[str, Any]]) -> bytes:
        """Export audit logs as CSV."""
        if not rows:
            return b""
        
        output = StringIO()
        fieldnames = rows[0].keys()
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        
        writer.writeheader()
        for row in rows:
            # Convert datetime objects to ISO strings
            row_data = dict(row)
            for key, value in row_data.items():
                if isinstance(value, datetime):
                    row_data[key] = value.isoformat()
                elif isinstance(value, dict):
                    row_data[key] = json.dumps(value)
            writer.writerow(row_data)
        
        return output.getvalue().encode('utf-8')

    def _export_as_xml(self, rows: List[Dict[str, Any]]) -> bytes:
        """Export audit logs as XML."""
        root = ET.Element("audit_logs")
        
        for row in rows:
            log_element = ET.SubElement(root, "audit_log")
            for key, value in row.items():
                field_element = ET.SubElement(log_element, key)
                if isinstance(value, datetime):
                    field_element.text = value.isoformat()
                elif isinstance(value, dict):
                    field_element.text = json.dumps(value)
                else:
                    field_element.text = str(value) if value is not None else ""
        
        return ET.tostring(root, encoding='utf-8', xml_declaration=True)


__all__ = ["AuditService", "AuditServiceError"]