"""
Application service orchestrating facet discovery and caching.

Responsibilities:
- Check cache before database query
- Coordinate between facet service and cache
- Handle cache invalidation logic
- Provide high-level facet operations
"""

import structlog
from typing import List, Optional
from datetime import datetime

from app.domain.entities.facet import FacetMetadata, FacetFieldName
from app.domain.interfaces import IFacetService, ICacheService
from app.domain.value_objects import TenantId

logger = structlog.get_logger(__name__)


class FacetApplicationService:
    """Application service orchestrating facet discovery and caching."""

    CACHE_TTL_SECONDS = 3600  # 1 hour
    CACHE_PREFIX = "facet:metadata"

    def __init__(
        self,
        facet_service: IFacetService,
        cache_service: ICacheService
    ):
        self.facet_service = facet_service
        self.cache = cache_service

    def _build_cache_key(self, tenant_id: TenantId) -> str:
        """Build cache key for tenant facets."""
        return f"{self.CACHE_PREFIX}:{tenant_id.value}"

    async def get_facets(
        self,
        tenant_id: TenantId,
        include_fields: Optional[List[FacetFieldName]] = None,
        force_refresh: bool = False
    ) -> FacetMetadata:
        """
        Get facet metadata with caching.

        Workflow:
        1. Check cache (unless force_refresh)
        2. If miss, generate from database
        3. Cache result
        4. Return to caller
        """
        # Try cache first
        if not force_refresh:
            cache_key = self._build_cache_key(tenant_id)
            cached_data = await self.cache.get(cache_key)

            if cached_data:
                try:
                    facets = self._deserialize_facets(cached_data, tenant_id)
                    if not facets.is_stale(max_age_seconds=self.CACHE_TTL_SECONDS):
                        logger.info("Returning cached facets", tenant_id=str(tenant_id.value))
                        return facets
                except Exception as e:
                    logger.warning("Failed to deserialize cached facets", error=str(e))

        # Cache miss or stale - regenerate
        logger.info("Generating fresh facets", tenant_id=str(tenant_id.value))

        include_fields_list = [f.value for f in include_fields] if include_fields else None
        facets = await self.facet_service.generate_facets(
            tenant_id=tenant_id,
            include_fields=include_fields_list,
            force_refresh=force_refresh
        )

        # Cache the result
        await self._cache_facets(facets)

        return facets

    async def _cache_facets(self, facet_metadata: FacetMetadata) -> None:
        """Cache facet metadata."""
        cache_key = self._build_cache_key(facet_metadata.tenant_id)

        try:
            serialized = self._serialize_facets(facet_metadata)
            await self.cache.set(
                key=cache_key,
                value=serialized,
                ttl=self.CACHE_TTL_SECONDS
            )

            logger.info(
                "Cached facet metadata",
                tenant_id=str(facet_metadata.tenant_id.value),
                ttl_seconds=self.CACHE_TTL_SECONDS
            )

        except Exception as e:
            logger.error("Failed to cache facets", error=str(e))

    async def invalidate_facets(self, tenant_id: TenantId) -> None:
        """
        Invalidate facet cache for tenant.

        Call this when:
        - Bulk profiles uploaded
        - Profiles deleted
        - Profile data significantly changed
        """
        logger.info("Invalidating facet cache", tenant_id=str(tenant_id.value))
        cache_key = self._build_cache_key(tenant_id)
        await self.cache.delete(cache_key)

    async def refresh_facets(self, tenant_id: TenantId) -> FacetMetadata:
        """Force refresh facets (bypass cache)."""
        return await self.get_facets(tenant_id, force_refresh=True)

    async def get_facet_statistics(self, tenant_id: TenantId) -> dict:
        """Get facet generation statistics."""
        return await self.facet_service.get_facet_statistics(tenant_id)

    def _serialize_facets(self, facet_metadata: FacetMetadata) -> dict:
        """Serialize FacetMetadata to dict for caching."""
        from app.domain.entities.facet import FacetType, FacetFieldName

        return {
            "tenant_id": str(facet_metadata.tenant_id.value),
            "facet_fields": [
                {
                    "field_name": f.field_name.value,
                    "facet_type": f.facet_type.value,
                    "display_name": f.display_name,
                    "description": f.description,
                    "values": [
                        {
                            "value": v.value,
                            "count": v.count,
                            "display_name": v.display_name,
                            "percentage": v.percentage,
                        }
                        for v in f.values
                    ],
                    "buckets": [
                        {
                            "label": b.label,
                            "min_value": b.min_value,
                            "max_value": b.max_value,
                            "count": b.count,
                            "percentage": b.percentage,
                        }
                        for b in f.buckets
                    ],
                    "min_value": f.min_value,
                    "max_value": f.max_value,
                    "searchable": f.searchable,
                    "multi_select": f.multi_select,
                    "total_count": f.total_count,
                    "unique_count": f.unique_count,
                }
                for f in facet_metadata.facet_fields
            ],
            "total_profiles": facet_metadata.total_profiles,
            "active_profiles": facet_metadata.active_profiles,
            "generated_at": facet_metadata.generated_at.isoformat(),
        }

    def _deserialize_facets(self, data: dict, tenant_id: TenantId) -> FacetMetadata:
        """Deserialize dict back to FacetMetadata."""
        from app.domain.entities.facet import (
            FacetField, FacetValue, RangeBucket, FacetType, FacetFieldName
        )

        facet_fields = []
        for field_data in data.get("facet_fields", []):
            values = [
                FacetValue(**v) for v in field_data.get("values", [])
            ]
            buckets = [
                RangeBucket(**b) for b in field_data.get("buckets", [])
            ]

            facet_fields.append(FacetField(
                field_name=FacetFieldName(field_data["field_name"]),
                facet_type=FacetType(field_data["facet_type"]),
                display_name=field_data["display_name"],
                description=field_data.get("description"),
                values=values,
                buckets=buckets,
                min_value=field_data.get("min_value"),
                max_value=field_data.get("max_value"),
                searchable=field_data.get("searchable", False),
                multi_select=field_data.get("multi_select", True),
                total_count=field_data.get("total_count", 0),
                unique_count=field_data.get("unique_count", 0),
            ))

        return FacetMetadata(
            tenant_id=tenant_id,
            facet_fields=facet_fields,
            total_profiles=data["total_profiles"],
            active_profiles=data["active_profiles"],
            generated_at=datetime.fromisoformat(data["generated_at"])
        )
