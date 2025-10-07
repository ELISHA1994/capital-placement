"""
Circuit breaker service for webhook endpoints.

This module implements the circuit breaker pattern to prevent cascading failures
and provide fault tolerance for webhook deliveries. It tracks endpoint health
and automatically opens circuits when failure thresholds are exceeded.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

import structlog

from app.domain.interfaces import IWebhookCircuitBreakerService, IDatabase
from app.api.schemas.webhook_schemas import CircuitBreakerState

logger = structlog.get_logger(__name__)


class WebhookCircuitBreakerService(IWebhookCircuitBreakerService):
    """Circuit breaker service for webhook endpoint reliability."""

    def __init__(self, database: IDatabase):
        """
        Initialize circuit breaker service.
        
        Args:
            database: Database interface for persistence
        """
        self.database = database
        self._circuit_states: Dict[str, Dict[str, Any]] = {}
        self._lock = asyncio.Lock()
        
    async def check_health(self) -> Dict[str, Any]:
        """Return circuit breaker service health."""
        async with self._lock:
            active_circuits = len(self._circuit_states)
            open_circuits = sum(
                1 for state in self._circuit_states.values() 
                if state.get("state") == CircuitBreakerState.OPEN
            )
            
        return {
            "status": "healthy",
            "service": "WebhookCircuitBreakerService",
            "active_circuits": active_circuits,
            "open_circuits": open_circuits,
            "cache_size": active_circuits
        }
    
    async def should_allow_request(self, endpoint_id: str) -> bool:
        """
        Check if requests should be allowed to an endpoint.
        
        Args:
            endpoint_id: Webhook endpoint ID
            
        Returns:
            True if requests are allowed
        """
        try:
            circuit_state = await self._get_circuit_state(endpoint_id)
            
            # If circuit is closed, allow requests
            if circuit_state["state"] == CircuitBreakerState.CLOSED:
                return True
            
            # If circuit is open, check if it's time to try half-open
            if circuit_state["state"] == CircuitBreakerState.OPEN:
                recovery_time = circuit_state.get("opened_at")
                if recovery_time:
                    recovery_timeout = circuit_state.get("recovery_timeout_seconds", 60)
                    if datetime.utcnow() >= recovery_time + timedelta(seconds=recovery_timeout):
                        # Move to half-open state
                        await self._update_circuit_state(
                            endpoint_id, 
                            CircuitBreakerState.HALF_OPEN,
                            {"half_open_attempts": 0}
                        )
                        logger.info(
                            "Circuit breaker moved to half-open",
                            endpoint_id=endpoint_id
                        )
                        return True
                return False
            
            # If circuit is half-open, allow limited requests
            if circuit_state["state"] == CircuitBreakerState.HALF_OPEN:
                max_half_open_calls = circuit_state.get("max_half_open_calls", 3)
                current_attempts = circuit_state.get("half_open_attempts", 0)
                
                if current_attempts < max_half_open_calls:
                    # Allow the request and increment attempt counter
                    await self._update_circuit_state(
                        endpoint_id,
                        CircuitBreakerState.HALF_OPEN,
                        {"half_open_attempts": current_attempts + 1}
                    )
                    return True
                return False
            
            # Default to allowing requests
            return True
            
        except Exception as e:
            logger.error(
                "Error checking circuit breaker state",
                endpoint_id=endpoint_id,
                error=str(e)
            )
            # Fail open - allow requests if there's an error
            return True
    
    async def record_success(self, endpoint_id: str, response_time_ms: int) -> None:
        """
        Record a successful webhook delivery.
        
        Args:
            endpoint_id: Webhook endpoint ID
            response_time_ms: Response time in milliseconds
        """
        try:
            circuit_state = await self._get_circuit_state(endpoint_id)
            
            # Update success metrics
            total_calls = circuit_state.get("total_calls", 0) + 1
            successful_calls = circuit_state.get("successful_calls", 0) + 1
            
            # Update average response time
            current_avg = circuit_state.get("avg_response_time_ms", 0)
            new_avg = ((current_avg * (successful_calls - 1)) + response_time_ms) / successful_calls
            
            updates = {
                "total_calls": total_calls,
                "successful_calls": successful_calls,
                "avg_response_time_ms": new_avg,
                "last_success_at": datetime.utcnow(),
                "failure_count": 0  # Reset consecutive failure count
            }
            
            # Handle half-open to closed transition
            if circuit_state["state"] == CircuitBreakerState.HALF_OPEN:
                half_open_successes = circuit_state.get("half_open_successes", 0) + 1
                updates["half_open_successes"] = half_open_successes
                
                # If enough successes in half-open, close the circuit
                min_successes = circuit_state.get("min_half_open_successes", 2)
                if half_open_successes >= min_successes:
                    updates["state"] = CircuitBreakerState.CLOSED
                    updates["half_open_attempts"] = 0
                    updates["half_open_successes"] = 0
                    logger.info(
                        "Circuit breaker closed after successful half-open period",
                        endpoint_id=endpoint_id,
                        half_open_successes=half_open_successes
                    )
            
            await self._update_circuit_state(endpoint_id, circuit_state["state"], updates)
            
            # Update database statistics
            await self._update_endpoint_stats(endpoint_id, {
                "total_deliveries": total_calls,
                "successful_deliveries": successful_calls,
                "avg_response_time_ms": new_avg,
                "last_successful_delivery": datetime.utcnow()
            })
            
        except Exception as e:
            logger.error(
                "Error recording circuit breaker success",
                endpoint_id=endpoint_id,
                response_time_ms=response_time_ms,
                error=str(e)
            )
    
    async def record_failure(self, endpoint_id: str, failure_reason: str) -> None:
        """
        Record a failed webhook delivery.
        
        Args:
            endpoint_id: Webhook endpoint ID
            failure_reason: Reason for failure
        """
        try:
            circuit_state = await self._get_circuit_state(endpoint_id)
            
            # Update failure metrics
            total_calls = circuit_state.get("total_calls", 0) + 1
            failed_calls = circuit_state.get("failed_calls", 0) + 1
            failure_count = circuit_state.get("failure_count", 0) + 1
            
            updates = {
                "total_calls": total_calls,
                "failed_calls": failed_calls,
                "failure_count": failure_count,
                "last_failure_at": datetime.utcnow(),
                "last_failure_reason": failure_reason
            }
            
            # Check if we should open the circuit
            failure_threshold = circuit_state.get("failure_threshold", 5)
            new_state = circuit_state["state"]
            
            if (circuit_state["state"] == CircuitBreakerState.CLOSED and 
                failure_count >= failure_threshold):
                # Open the circuit
                new_state = CircuitBreakerState.OPEN
                updates["state"] = new_state
                updates["opened_at"] = datetime.utcnow()
                
                logger.warning(
                    "Circuit breaker opened due to consecutive failures",
                    endpoint_id=endpoint_id,
                    failure_count=failure_count,
                    failure_threshold=failure_threshold,
                    failure_reason=failure_reason
                )
            
            elif circuit_state["state"] == CircuitBreakerState.HALF_OPEN:
                # Failure in half-open state, go back to open
                new_state = CircuitBreakerState.OPEN
                updates["state"] = new_state
                updates["opened_at"] = datetime.utcnow()
                updates["half_open_attempts"] = 0
                updates["half_open_successes"] = 0
                
                logger.warning(
                    "Circuit breaker reopened after half-open failure",
                    endpoint_id=endpoint_id,
                    failure_reason=failure_reason
                )
            
            await self._update_circuit_state(endpoint_id, new_state, updates)
            
            # Update database statistics
            await self._update_endpoint_stats(endpoint_id, {
                "total_deliveries": total_calls,
                "failed_deliveries": failed_calls,
                "failure_count": failure_count
            })
            
        except Exception as e:
            logger.error(
                "Error recording circuit breaker failure",
                endpoint_id=endpoint_id,
                failure_reason=failure_reason,
                error=str(e)
            )
    
    async def get_circuit_state(self, endpoint_id: str) -> Dict[str, Any]:
        """
        Get circuit breaker state for an endpoint.
        
        Args:
            endpoint_id: Webhook endpoint ID
            
        Returns:
            Circuit breaker state information
        """
        try:
            return await self._get_circuit_state(endpoint_id)
        except Exception as e:
            logger.error(
                "Error getting circuit state",
                endpoint_id=endpoint_id,
                error=str(e)
            )
            return {
                "state": CircuitBreakerState.CLOSED,
                "error": str(e)
            }
    
    async def force_circuit_state(
        self,
        endpoint_id: str,
        state: str,
        *,
        admin_user_id: Optional[str] = None,
        reason: Optional[str] = None
    ) -> bool:
        """
        Manually set circuit breaker state.
        
        Args:
            endpoint_id: Webhook endpoint ID
            state: Target state (open, closed, half_open)
            admin_user_id: Admin user making the change
            reason: Reason for manual override
            
        Returns:
            True if state change was successful
        """
        try:
            # Validate state
            if state not in [s.value for s in CircuitBreakerState]:
                logger.error(
                    "Invalid circuit breaker state",
                    endpoint_id=endpoint_id,
                    requested_state=state,
                    valid_states=[s.value for s in CircuitBreakerState]
                )
                return False
            
            circuit_state_enum = CircuitBreakerState(state)
            
            # Prepare state updates
            updates = {
                "state": circuit_state_enum,
                "manual_override": True,
                "override_by": admin_user_id,
                "override_reason": reason,
                "override_at": datetime.utcnow()
            }
            
            # State-specific updates
            if circuit_state_enum == CircuitBreakerState.OPEN:
                updates["opened_at"] = datetime.utcnow()
            elif circuit_state_enum == CircuitBreakerState.HALF_OPEN:
                updates["half_open_attempts"] = 0
                updates["half_open_successes"] = 0
            elif circuit_state_enum == CircuitBreakerState.CLOSED:
                updates["failure_count"] = 0
                updates["half_open_attempts"] = 0
                updates["half_open_successes"] = 0
            
            await self._update_circuit_state(endpoint_id, circuit_state_enum, updates)
            
            logger.info(
                "Circuit breaker state manually changed",
                endpoint_id=endpoint_id,
                new_state=state,
                admin_user_id=admin_user_id,
                reason=reason
            )
            
            return True
            
        except Exception as e:
            logger.error(
                "Error forcing circuit breaker state",
                endpoint_id=endpoint_id,
                requested_state=state,
                admin_user_id=admin_user_id,
                error=str(e)
            )
            return False
    
    async def _get_circuit_state(self, endpoint_id: str) -> Dict[str, Any]:
        """Get circuit state from cache or database."""
        async with self._lock:
            # Check cache first
            if endpoint_id in self._circuit_states:
                return self._circuit_states[endpoint_id]
        
        # Load from database
        try:
            endpoint_data = await self.database.get_item(
                "webhook_endpoints",
                endpoint_id,
                endpoint_id
            )
            
            if endpoint_data:
                # Extract circuit breaker state
                state_data = {
                    "state": CircuitBreakerState(endpoint_data.get("circuit_state", "closed")),
                    "failure_count": endpoint_data.get("failure_count", 0),
                    "total_calls": endpoint_data.get("total_deliveries", 0),
                    "successful_calls": endpoint_data.get("successful_deliveries", 0),
                    "failed_calls": endpoint_data.get("failed_deliveries", 0),
                    "avg_response_time_ms": endpoint_data.get("avg_response_time_ms"),
                    "last_failure_at": endpoint_data.get("last_failure_at"),
                    "last_success_at": endpoint_data.get("last_successful_delivery"),
                    "opened_at": endpoint_data.get("circuit_opened_at"),
                    "failure_threshold": endpoint_data.get("retry_policy", {}).get("failure_threshold", 5),
                    "recovery_timeout_seconds": endpoint_data.get("retry_policy", {}).get("recovery_timeout_seconds", 60),
                    "max_half_open_calls": endpoint_data.get("retry_policy", {}).get("half_open_max_calls", 3),
                    "min_half_open_successes": 2,
                    "half_open_attempts": endpoint_data.get("half_open_attempts", 0),
                    "half_open_successes": 0
                }
            else:
                # Default state for new endpoint
                state_data = {
                    "state": CircuitBreakerState.CLOSED,
                    "failure_count": 0,
                    "total_calls": 0,
                    "successful_calls": 0,
                    "failed_calls": 0,
                    "failure_threshold": 5,
                    "recovery_timeout_seconds": 60,
                    "max_half_open_calls": 3,
                    "min_half_open_successes": 2,
                    "half_open_attempts": 0,
                    "half_open_successes": 0
                }
            
            # Cache the state
            async with self._lock:
                self._circuit_states[endpoint_id] = state_data
            
            return state_data
            
        except Exception as e:
            logger.error(
                "Error loading circuit state from database",
                endpoint_id=endpoint_id,
                error=str(e)
            )
            # Return default closed state on error
            return {
                "state": CircuitBreakerState.CLOSED,
                "failure_count": 0,
                "total_calls": 0,
                "successful_calls": 0,
                "failed_calls": 0,
                "failure_threshold": 5,
                "recovery_timeout_seconds": 60,
                "max_half_open_calls": 3,
                "min_half_open_successes": 2,
                "half_open_attempts": 0,
                "half_open_successes": 0
            }
    
    async def _update_circuit_state(
        self,
        endpoint_id: str,
        state: CircuitBreakerState,
        updates: Dict[str, Any]
    ) -> None:
        """Update circuit state in cache and database."""
        try:
            # Update cache
            async with self._lock:
                if endpoint_id not in self._circuit_states:
                    self._circuit_states[endpoint_id] = {}
                
                self._circuit_states[endpoint_id].update(updates)
                self._circuit_states[endpoint_id]["state"] = state
            
            # Update database (non-blocking)
            asyncio.create_task(self._persist_circuit_state(endpoint_id, updates))
            
        except Exception as e:
            logger.error(
                "Error updating circuit state",
                endpoint_id=endpoint_id,
                state=state.value,
                error=str(e)
            )
    
    async def _persist_circuit_state(self, endpoint_id: str, updates: Dict[str, Any]) -> None:
        """Persist circuit state to database."""
        try:
            # Convert datetime objects to strings for JSON serialization
            db_updates = {}
            for key, value in updates.items():
                if isinstance(value, datetime):
                    db_updates[key] = value.isoformat()
                elif isinstance(value, CircuitBreakerState):
                    db_updates[f"circuit_{key}"] = value.value
                else:
                    db_updates[key] = value
            
            await self.database.update_item(
                "webhook_endpoints",
                endpoint_id,
                db_updates
            )
            
        except Exception as e:
            logger.warning(
                "Failed to persist circuit state to database",
                endpoint_id=endpoint_id,
                error=str(e)
            )
    
    async def _update_endpoint_stats(self, endpoint_id: str, stats: Dict[str, Any]) -> None:
        """Update endpoint statistics in database."""
        try:
            # Convert datetime objects for JSON serialization
            db_stats = {}
            for key, value in stats.items():
                if isinstance(value, datetime):
                    db_stats[key] = value.isoformat()
                else:
                    db_stats[key] = value
            
            await self.database.update_item(
                "webhook_endpoints",
                endpoint_id,
                db_stats
            )
            
        except Exception as e:
            logger.warning(
                "Failed to update endpoint statistics",
                endpoint_id=endpoint_id,
                error=str(e)
            )


__all__ = ["WebhookCircuitBreakerService"]