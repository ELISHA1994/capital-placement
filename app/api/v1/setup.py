"""
System Setup API Endpoints

Provides initial system setup functionality:
- System status check
- Super admin creation (bootstrap)
- System initialization
"""

from typing import Dict, Any, Optional

import structlog
from fastapi import APIRouter, HTTPException, status, Depends
from pydantic import BaseModel, Field

from app.core.dependencies import get_async_bootstrap_service
from app.services.bootstrap_service import BootstrapService

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/setup", tags=["setup"])


class SuperAdminCreateRequest(BaseModel):
    """Request model for creating super admin"""
    email: str = Field(..., description="Super admin email address")
    password: str = Field(..., min_length=8, description="Super admin password")
    full_name: str = Field(..., description="Super admin full name")


class SetupStatusResponse(BaseModel):
    """Response model for setup status"""
    initialized: bool
    has_system_tenant: bool
    has_super_admin: bool
    tenant_count: int
    total_users: int
    system_tenant_id: Optional[str] = None
    message: str


class SuperAdminCreateResponse(BaseModel):
    """Response model for super admin creation"""
    user: Dict[str, Any]
    message: str


@router.get(
    "/status",
    response_model=SetupStatusResponse,
    summary="Get system setup status",
    description="Check the current initialization status of the system"
)
async def get_setup_status(
    bootstrap_service: BootstrapService = Depends(get_async_bootstrap_service)
):
    """Get system setup and initialization status"""
    
    try:
        status_info = await bootstrap_service.get_system_status()
        
        # Add a descriptive message
        if status_info["initialized"]:
            message = "System is fully initialized and ready for use"
        elif status_info["has_system_tenant"] and not status_info["has_super_admin"]:
            message = "System tenant exists but no super admin found. Use /initialize to create one"
        else:
            message = "System requires initialization. Use /initialize to setup"
        
        return SetupStatusResponse(
            **status_info,
            message=message
        )
        
    except Exception as e:
        logger.error(f"Failed to get setup status: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get setup status: {str(e)}"
        )


@router.post(
    "/initialize",
    response_model=SuperAdminCreateResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Initialize system with super admin",
    description="Create the first super admin user and initialize the system. Only works if no super admin exists."
)
async def initialize_system(
    admin_data: SuperAdminCreateRequest,
    bootstrap_service: BootstrapService = Depends(get_async_bootstrap_service)
):
    """Initialize the system by creating the first super admin user"""
    
    try:
        # Check if already initialized
        if await bootstrap_service.has_super_admin():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="System already initialized. Super admin already exists."
            )
        
        # Create super admin
        result = await bootstrap_service.create_super_admin(
            email=admin_data.email,
            password=admin_data.password,
            full_name=admin_data.full_name
        )
        
        logger.info(
            "System initialized successfully",
            email=admin_data.email
        )
        
        return SuperAdminCreateResponse(**result)
        
    except ValueError as e:
        logger.warning(f"System initialization failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"System initialization error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"System initialization failed: {str(e)}"
        )
