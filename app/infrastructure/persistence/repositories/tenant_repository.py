"""PostgreSQL implementation of ITenantRepository using TenantMapper."""

from __future__ import annotations

from typing import List, Optional
from uuid import UUID

from app.domain.entities.tenant import Tenant, TenantStatus, SubscriptionTier
from app.domain.repositories.tenant_repository import ITenantRepository
from app.domain.value_objects import TenantId
from app.infrastructure.persistence.mappers.tenant_mapper import TenantMapper
from app.models.tenant_models import TenantTable, TenantConfigurationTable
from app.infrastructure.providers.postgres_provider import get_postgres_adapter


class PostgresTenantRepository(ITenantRepository):
    """PostgreSQL adapter implementation of ITenantRepository."""

    def __init__(self):
        self._adapter = None

    async def _get_adapter(self):
        """Get database adapter (lazy initialization)."""
        if self._adapter is None:
            self._adapter = await get_postgres_adapter()
        return self._adapter

    async def save(self, tenant: Tenant) -> Tenant:
        """Save tenant to database and return updated domain entity."""
        adapter = await self._get_adapter()
        
        try:
            # Check if tenant exists
            existing = await self.find_by_id(tenant.id)
            
            tenant_table, config_table = TenantMapper.to_persistence(tenant)
            
            if existing:
                # Update existing tenant
                await adapter.execute(
                    """
                    UPDATE tenants 
                    SET name = $2, slug = $3, display_name = $4, description = $5,
                        subscription_tier = $6, is_active = $7, is_system_tenant = $8,
                        primary_contact_email = $9, data_region = $10, timezone = $11,
                        locale = $12, date_format = $13, updated_at = $14
                    WHERE id = $1
                    """,
                    tenant_table.id, tenant_table.name, tenant_table.slug,
                    tenant_table.display_name, tenant_table.description,
                    tenant_table.subscription_tier, tenant_table.is_active,
                    tenant_table.is_system_tenant, tenant_table.primary_contact_email,
                    tenant_table.data_region, tenant_table.timezone, tenant_table.locale,
                    tenant_table.date_format, tenant_table.updated_at
                )
                
                # Update or insert configuration
                config_exists = await adapter.fetch_one(
                    "SELECT id FROM tenant_configurations WHERE id = $1",
                    config_table.id
                )
                
                if config_exists:
                    await self._update_config_table(adapter, config_table)
                else:
                    await self._insert_config_table(adapter, config_table)
                
            else:
                # Insert new tenant
                await adapter.execute(
                    """
                    INSERT INTO tenants (
                        id, name, slug, display_name, description, subscription_tier,
                        is_active, is_system_tenant, primary_contact_email, data_region,
                        timezone, locale, date_format, created_at, updated_at
                    ) VALUES (
                        $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15
                    )
                    """,
                    tenant_table.id, tenant_table.name, tenant_table.slug,
                    tenant_table.display_name, tenant_table.description,
                    tenant_table.subscription_tier, tenant_table.is_active,
                    tenant_table.is_system_tenant, tenant_table.primary_contact_email,
                    tenant_table.data_region, tenant_table.timezone, tenant_table.locale,
                    tenant_table.date_format, tenant_table.created_at, tenant_table.updated_at
                )
                
                # Insert configuration
                await self._insert_config_table(adapter, config_table)
            
            return tenant
            
        except Exception as e:
            raise Exception(f"Failed to save tenant: {str(e)}")

    async def _insert_config_table(self, adapter, config_table: TenantConfigurationTable):
        """Insert tenant configuration table."""
        await adapter.execute(
            """
            INSERT INTO tenant_configurations (
                id, name, display_name, description, subscription_tier,
                subscription_start_date, subscription_end_date, billing_configuration,
                search_configuration, processing_configuration, quota_limits,
                feature_flags, index_strategy, dedicated_search_index, data_region,
                is_active, is_suspended, suspension_reason, is_system_tenant,
                current_usage, primary_contact_email, billing_contact_email,
                technical_contact_email, timezone, locale, date_format,
                created_at, updated_at
            ) VALUES (
                $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15,
                $16, $17, $18, $19, $20, $21, $22, $23, $24, $25, $26, $27, $28
            )
            """,
            config_table.id, config_table.name, config_table.display_name,
            config_table.description, config_table.subscription_tier,
            config_table.subscription_start_date, config_table.subscription_end_date,
            config_table.billing_configuration, config_table.search_configuration,
            config_table.processing_configuration, config_table.quota_limits,
            config_table.feature_flags, config_table.index_strategy,
            config_table.dedicated_search_index, config_table.data_region,
            config_table.is_active, config_table.is_suspended,
            config_table.suspension_reason, config_table.is_system_tenant,
            config_table.current_usage, config_table.primary_contact_email,
            config_table.billing_contact_email, config_table.technical_contact_email,
            config_table.timezone, config_table.locale, config_table.date_format,
            config_table.created_at, config_table.updated_at
        )

    async def _update_config_table(self, adapter, config_table: TenantConfigurationTable):
        """Update tenant configuration table."""
        await adapter.execute(
            """
            UPDATE tenant_configurations 
            SET name = $2, display_name = $3, description = $4, subscription_tier = $5,
                subscription_start_date = $6, subscription_end_date = $7,
                billing_configuration = $8, search_configuration = $9,
                processing_configuration = $10, quota_limits = $11, feature_flags = $12,
                index_strategy = $13, dedicated_search_index = $14, data_region = $15,
                is_active = $16, is_suspended = $17, suspension_reason = $18,
                is_system_tenant = $19, current_usage = $20, primary_contact_email = $21,
                billing_contact_email = $22, technical_contact_email = $23,
                timezone = $24, locale = $25, date_format = $26, updated_at = $27
            WHERE id = $1
            """,
            config_table.id, config_table.name, config_table.display_name,
            config_table.description, config_table.subscription_tier,
            config_table.subscription_start_date, config_table.subscription_end_date,
            config_table.billing_configuration, config_table.search_configuration,
            config_table.processing_configuration, config_table.quota_limits,
            config_table.feature_flags, config_table.index_strategy,
            config_table.dedicated_search_index, config_table.data_region,
            config_table.is_active, config_table.is_suspended,
            config_table.suspension_reason, config_table.is_system_tenant,
            config_table.current_usage, config_table.primary_contact_email,
            config_table.billing_contact_email, config_table.technical_contact_email,
            config_table.timezone, config_table.locale, config_table.date_format,
            config_table.updated_at
        )

    async def find_by_id(self, tenant_id: TenantId) -> Optional[Tenant]:
        """Find tenant by ID."""
        adapter = await self._get_adapter()
        
        try:
            # Get basic tenant record
            tenant_record = await adapter.fetch_one(
                "SELECT * FROM tenants WHERE id = $1",
                tenant_id.value
            )
            
            if not tenant_record:
                return None
            
            # Get configuration record
            config_record = await adapter.fetch_one(
                "SELECT * FROM tenant_configurations WHERE id = $1",
                tenant_id.value
            )
            
            tenant_table = TenantTable(**dict(tenant_record))
            config_table = None
            if config_record:
                config_table = TenantConfigurationTable(**dict(config_record))
            
            return TenantMapper.to_domain(tenant_table, config_table)
            
        except Exception as e:
            raise Exception(f"Failed to find tenant by ID: {str(e)}")

    async def find_by_name(self, name: str) -> Optional[Tenant]:
        """Find tenant by name."""
        adapter = await self._get_adapter()
        
        try:
            tenant_record = await adapter.fetch_one(
                "SELECT * FROM tenants WHERE name = $1",
                name
            )
            
            if not tenant_record:
                return None
            
            config_record = await adapter.fetch_one(
                "SELECT * FROM tenant_configurations WHERE name = $1",
                name
            )
            
            tenant_table = TenantTable(**dict(tenant_record))
            config_table = None
            if config_record:
                config_table = TenantConfigurationTable(**dict(config_record))
            
            return TenantMapper.to_domain(tenant_table, config_table)
            
        except Exception as e:
            raise Exception(f"Failed to find tenant by name: {str(e)}")

    async def find_all(self, limit: int = 100, offset: int = 0) -> List[Tenant]:
        """Find all tenants with pagination."""
        adapter = await self._get_adapter()
        
        try:
            tenant_records = await adapter.fetch_all(
                "SELECT * FROM tenants ORDER BY created_at DESC LIMIT $1 OFFSET $2",
                limit, offset
            )
            
            tenants = []
            for tenant_record in tenant_records:
                tenant_table = TenantTable(**dict(tenant_record))
                
                # Get config for this tenant
                config_record = await adapter.fetch_one(
                    "SELECT * FROM tenant_configurations WHERE id = $1",
                    tenant_table.id
                )
                
                config_table = None
                if config_record:
                    config_table = TenantConfigurationTable(**dict(config_record))
                
                tenants.append(TenantMapper.to_domain(tenant_table, config_table))
            
            return tenants
            
        except Exception as e:
            raise Exception(f"Failed to find all tenants: {str(e)}")

    async def find_by_status(self, status: TenantStatus) -> List[Tenant]:
        """Find tenants by status."""
        adapter = await self._get_adapter()
        
        try:
            is_active = status == TenantStatus.ACTIVE
            
            tenant_records = await adapter.fetch_all(
                "SELECT * FROM tenants WHERE is_active = $1 ORDER BY created_at DESC",
                is_active
            )
            
            tenants = []
            for tenant_record in tenant_records:
                tenant_table = TenantTable(**dict(tenant_record))
                
                config_record = await adapter.fetch_one(
                    "SELECT * FROM tenant_configurations WHERE id = $1",
                    tenant_table.id
                )
                
                config_table = None
                if config_record:
                    config_table = TenantConfigurationTable(**dict(config_record))
                
                tenants.append(TenantMapper.to_domain(tenant_table, config_table))
            
            return tenants
            
        except Exception as e:
            raise Exception(f"Failed to find tenants by status: {str(e)}")

    async def find_by_subscription_tier(self, tier: SubscriptionTier) -> List[Tenant]:
        """Find tenants by subscription tier."""
        adapter = await self._get_adapter()
        
        try:
            tenant_records = await adapter.fetch_all(
                "SELECT * FROM tenants WHERE subscription_tier = $1 ORDER BY created_at DESC",
                tier.value
            )
            
            tenants = []
            for tenant_record in tenant_records:
                tenant_table = TenantTable(**dict(tenant_record))
                
                config_record = await adapter.fetch_one(
                    "SELECT * FROM tenant_configurations WHERE id = $1",
                    tenant_table.id
                )
                
                config_table = None
                if config_record:
                    config_table = TenantConfigurationTable(**dict(config_record))
                
                tenants.append(TenantMapper.to_domain(tenant_table, config_table))
            
            return tenants
            
        except Exception as e:
            raise Exception(f"Failed to find tenants by subscription tier: {str(e)}")

    async def delete_by_id(self, tenant_id: TenantId) -> bool:
        """Delete tenant by ID (cascade delete)."""
        adapter = await self._get_adapter()
        
        try:
            # Delete configuration first
            await adapter.execute(
                "DELETE FROM tenant_configurations WHERE id = $1",
                tenant_id.value
            )
            
            # Delete tenant (will cascade to related records)
            result = await adapter.execute(
                "DELETE FROM tenants WHERE id = $1",
                tenant_id.value
            )
            
            return result and result.split()[-1] != '0'
            
        except Exception as e:
            raise Exception(f"Failed to delete tenant: {str(e)}")

    async def count_all(self) -> int:
        """Count all tenants."""
        adapter = await self._get_adapter()
        
        try:
            record = await adapter.fetch_one("SELECT COUNT(*) as count FROM tenants")
            return record['count'] if record else 0
            
        except Exception as e:
            raise Exception(f"Failed to count tenants: {str(e)}")

    async def find_system_tenant(self) -> Optional[Tenant]:
        """Find the system tenant."""
        adapter = await self._get_adapter()
        
        try:
            tenant_record = await adapter.fetch_one(
                "SELECT * FROM tenants WHERE is_system_tenant = true LIMIT 1"
            )
            
            if not tenant_record:
                return None
            
            config_record = await adapter.fetch_one(
                "SELECT * FROM tenant_configurations WHERE is_system_tenant = true LIMIT 1"
            )
            
            tenant_table = TenantTable(**dict(tenant_record))
            config_table = None
            if config_record:
                config_table = TenantConfigurationTable(**dict(config_record))
            
            return TenantMapper.to_domain(tenant_table, config_table)
            
        except Exception as e:
            raise Exception(f"Failed to find system tenant: {str(e)}")

    async def update_usage_metrics(self, tenant_id: TenantId, metrics_update: dict) -> None:
        """Update tenant usage metrics."""
        adapter = await self._get_adapter()
        
        try:
            # Get current usage
            config_record = await adapter.fetch_one(
                "SELECT current_usage FROM tenant_configurations WHERE id = $1",
                tenant_id.value
            )
            
            if config_record:
                current_usage = config_record['current_usage'] or {}
                current_usage.update(metrics_update)
                current_usage['metrics_updated_at'] = "NOW()"
                
                await adapter.execute(
                    "UPDATE tenant_configurations SET current_usage = $1, updated_at = NOW() WHERE id = $2",
                    current_usage, tenant_id.value
                )
            
        except Exception as e:
            raise Exception(f"Failed to update usage metrics: {str(e)}")


__all__ = ["PostgresTenantRepository"]