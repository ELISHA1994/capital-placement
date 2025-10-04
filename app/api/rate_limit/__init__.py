"""
API Dependencies Module

Provides FastAPI dependencies for:
- Rate limiting
- Authentication and authorization
- Request validation
- Resource management
"""

from .rate_limit_dependencies import (
    RateLimitServiceDep,
    UserRateLimitDep,
    IPRateLimitDep,
    UploadRateLimitDep,
    StrictRateLimitDep,
    RateLimitCheckerDep,
    get_client_ip,
    get_user_agent,
    is_admin_user,
    get_tenant_quota_limits,
    get_tenant_rate_limit_config,
    create_rate_limit_dependency,
)

__all__ = [
    "RateLimitServiceDep",
    "UserRateLimitDep", 
    "IPRateLimitDep",
    "UploadRateLimitDep",
    "StrictRateLimitDep",
    "RateLimitCheckerDep",
    "get_client_ip",
    "get_user_agent",
    "is_admin_user",
    "get_tenant_quota_limits",
    "get_tenant_rate_limit_config",
    "create_rate_limit_dependency",
]