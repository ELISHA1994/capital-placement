"""
PostgreSQL-based facet extraction using efficient aggregation queries.

Uses database-level aggregations for performance:
- COUNT(*) and GROUP BY for term facets
- CASE statements for range bucketing
- JSON aggregation for nested data (skills, education)
"""

import structlog
from typing import Dict, List, Optional, Any
from uuid import UUID

from app.domain.entities.facet import (
    FacetMetadata, FacetField, FacetValue, RangeBucket,
    FacetType, FacetFieldName
)
from app.domain.interfaces import IFacetService
from app.domain.value_objects import TenantId
from app.infrastructure.adapters.postgres_adapter import PostgresAdapter

logger = structlog.get_logger(__name__)


class PostgresFacetAdapter(IFacetService):
    """PostgreSQL-based facet extraction using efficient aggregation queries."""

    def __init__(self, db_adapter: PostgresAdapter):
        self.db = db_adapter

    async def check_health(self) -> Dict[str, Any]:
        """Health check for facet service."""
        return {"status": "healthy", "service": "facet_adapter"}

    async def generate_facets(
        self,
        tenant_id: TenantId,
        include_fields: Optional[List[str]] = None,
        force_refresh: bool = False
    ) -> FacetMetadata:
        """Generate complete facet metadata by aggregating profile data."""
        logger.info("Generating facets", tenant_id=str(tenant_id.value))

        # Get total profile counts
        total_profiles, active_profiles = await self._get_profile_counts(tenant_id)

        # Build facet fields
        facet_fields: List[FacetField] = []

        # Determine which fields to include
        fields_to_include = include_fields or [
            FacetFieldName.SKILLS.value,
            FacetFieldName.EXPERIENCE_LEVEL.value,
            FacetFieldName.TOTAL_EXPERIENCE_YEARS.value,
            FacetFieldName.EDUCATION_LEVEL.value,
            FacetFieldName.LOCATION_COUNTRY.value,
            FacetFieldName.LOCATION_CITY.value,
            FacetFieldName.COMPANIES.value,
            FacetFieldName.LANGUAGES.value,
        ]

        # Generate each facet field
        for field_name_str in fields_to_include:
            try:
                field_name = FacetFieldName(field_name_str)
                facet_field = await self._generate_facet_field(
                    tenant_id, field_name, total_profiles
                )
                if facet_field:
                    facet_fields.append(facet_field)
            except Exception as e:
                logger.error(
                    "Failed to generate facet field",
                    field_name=field_name_str,
                    error=str(e)
                )

        return FacetMetadata(
            tenant_id=tenant_id,
            facet_fields=facet_fields,
            total_profiles=total_profiles,
            active_profiles=active_profiles
        )

    async def _get_profile_counts(self, tenant_id: TenantId) -> tuple[int, int]:
        """Get total and active profile counts."""
        query = """
            SELECT
                COUNT(*) as total,
                COUNT(*) FILTER (WHERE status = 'active') as active
            FROM profiles
            WHERE tenant_id = $1
              AND is_deleted = false
        """

        async with self.db.get_connection() as conn:
            row = await conn.fetchrow(query, tenant_id.value)
            return row['total'], row['active']

    async def _generate_facet_field(
        self,
        tenant_id: TenantId,
        field_name: FacetFieldName,
        total_profiles: int
    ) -> Optional[FacetField]:
        """Generate a specific facet field with values."""
        if field_name == FacetFieldName.SKILLS:
            return await self._generate_skills_facet(tenant_id, total_profiles)
        elif field_name == FacetFieldName.EXPERIENCE_LEVEL:
            return await self._generate_experience_level_facet(tenant_id, total_profiles)
        elif field_name == FacetFieldName.TOTAL_EXPERIENCE_YEARS:
            return await self._generate_experience_years_facet(tenant_id, total_profiles)
        elif field_name == FacetFieldName.EDUCATION_LEVEL:
            return await self._generate_education_level_facet(tenant_id, total_profiles)
        elif field_name in (FacetFieldName.LOCATION_COUNTRY, FacetFieldName.LOCATION_CITY):
            return await self._generate_location_facet(tenant_id, field_name, total_profiles)
        elif field_name == FacetFieldName.COMPANIES:
            return await self._generate_companies_facet(tenant_id, total_profiles)
        elif field_name == FacetFieldName.LANGUAGES:
            return await self._generate_languages_facet(tenant_id, total_profiles)
        return None

    async def _generate_skills_facet(
        self,
        tenant_id: TenantId,
        total_profiles: int
    ) -> FacetField:
        """Generate skills facet by unnesting and aggregating skill arrays."""
        query = """
            SELECT
                skill,
                COUNT(DISTINCT p.id) as profile_count
            FROM profiles p,
                 UNNEST(p.normalized_skills) AS skill
            WHERE p.tenant_id = $1
              AND p.status = 'active'
              AND p.is_deleted = false
              AND skill IS NOT NULL
              AND skill != ''
            GROUP BY skill
            ORDER BY profile_count DESC
            LIMIT 500
        """

        async with self.db.get_connection() as conn:
            rows = await conn.fetch(query, tenant_id.value)

        values = [
            FacetValue(
                value=row['skill'],
                count=row['profile_count'],
                display_name=row['skill'],
                percentage=round((row['profile_count'] / total_profiles) * 100, 2) if total_profiles > 0 else 0.0
            )
            for row in rows
        ]

        return FacetField(
            field_name=FacetFieldName.SKILLS,
            facet_type=FacetType.TERMS,
            display_name="Skills & Technologies",
            description="Technical and soft skills",
            values=values,
            searchable=True,
            multi_select=True,
            total_count=sum(v.count for v in values),
            unique_count=len(values)
        )

    async def _generate_experience_level_facet(
        self,
        tenant_id: TenantId,
        total_profiles: int
    ) -> FacetField:
        """Generate experience level categorical facet."""
        query = """
            SELECT
                experience_level,
                COUNT(*) as count
            FROM profiles
            WHERE tenant_id = $1
              AND status = 'active'
              AND is_deleted = false
              AND experience_level IS NOT NULL
            GROUP BY experience_level
            ORDER BY count DESC
        """

        async with self.db.get_connection() as conn:
            rows = await conn.fetch(query, tenant_id.value)

        display_names = {
            "entry": "Entry Level",
            "junior": "Junior",
            "mid": "Mid-Level",
            "senior": "Senior",
            "lead": "Lead",
            "principal": "Principal",
            "executive": "Executive"
        }

        values = [
            FacetValue(
                value=row['experience_level'],
                count=row['count'],
                display_name=display_names.get(row['experience_level'], row['experience_level']),
                percentage=round((row['count'] / total_profiles) * 100, 2) if total_profiles > 0 else 0.0
            )
            for row in rows
        ]

        return FacetField(
            field_name=FacetFieldName.EXPERIENCE_LEVEL,
            facet_type=FacetType.TERMS,
            display_name="Experience Level",
            description="Professional seniority level",
            values=values,
            searchable=False,
            multi_select=True,
            total_count=sum(v.count for v in values),
            unique_count=len(values)
        )

    async def _generate_experience_years_facet(
        self,
        tenant_id: TenantId,
        total_profiles: int
    ) -> FacetField:
        """Generate experience years range facet with bucketing."""
        query = """
            SELECT
                CASE
                    WHEN COALESCE((profile_data->>'total_experience_years')::float, 0) BETWEEN 0 AND 2 THEN '0-2'
                    WHEN COALESCE((profile_data->>'total_experience_years')::float, 0) BETWEEN 3 AND 5 THEN '3-5'
                    WHEN COALESCE((profile_data->>'total_experience_years')::float, 0) BETWEEN 6 AND 10 THEN '6-10'
                    WHEN COALESCE((profile_data->>'total_experience_years')::float, 0) >= 11 THEN '11+'
                    ELSE 'unknown'
                END as bucket,
                COUNT(*) as count
            FROM profiles
            WHERE tenant_id = $1
              AND status = 'active'
              AND is_deleted = false
            GROUP BY bucket
            ORDER BY bucket
        """

        async with self.db.get_connection() as conn:
            rows = await conn.fetch(query, tenant_id.value)

        bucket_map = {row['bucket']: row['count'] for row in rows}

        buckets = [
            RangeBucket(
                label="Entry Level (0-2 years)",
                min_value=0,
                max_value=2,
                count=bucket_map.get('0-2', 0),
                percentage=round((bucket_map.get('0-2', 0) / total_profiles) * 100, 2) if total_profiles > 0 else 0.0
            ),
            RangeBucket(
                label="Mid Level (3-5 years)",
                min_value=3,
                max_value=5,
                count=bucket_map.get('3-5', 0),
                percentage=round((bucket_map.get('3-5', 0) / total_profiles) * 100, 2) if total_profiles > 0 else 0.0
            ),
            RangeBucket(
                label="Senior Level (6-10 years)",
                min_value=6,
                max_value=10,
                count=bucket_map.get('6-10', 0),
                percentage=round((bucket_map.get('6-10', 0) / total_profiles) * 100, 2) if total_profiles > 0 else 0.0
            ),
            RangeBucket(
                label="Lead/Expert (11+ years)",
                min_value=11,
                max_value=None,
                count=bucket_map.get('11+', 0),
                percentage=round((bucket_map.get('11+', 0) / total_profiles) * 100, 2) if total_profiles > 0 else 0.0
            ),
        ]

        return FacetField(
            field_name=FacetFieldName.TOTAL_EXPERIENCE_YEARS,
            facet_type=FacetType.RANGE,
            display_name="Years of Experience",
            description="Total professional experience",
            buckets=buckets,
            min_value=0,
            max_value=40,
            searchable=False,
            multi_select=False,
            total_count=sum(b.count for b in buckets),
            unique_count=len(buckets)
        )

    async def _generate_education_level_facet(
        self,
        tenant_id: TenantId,
        total_profiles: int
    ) -> FacetField:
        """Generate education level facet by extracting from JSONB."""
        query = """
            WITH education_degrees AS (
                SELECT
                    p.id,
                    jsonb_array_elements(p.profile_data->'education')->>'degree' as degree
                FROM profiles p
                WHERE p.tenant_id = $1
                  AND p.status = 'active'
                  AND p.is_deleted = false
                  AND p.profile_data ? 'education'
            ),
            degree_levels AS (
                SELECT
                    CASE
                        WHEN degree ILIKE '%phd%' OR degree ILIKE '%doctorate%' THEN 'PhD/Doctorate'
                        WHEN degree ILIKE '%master%' OR degree ILIKE '%mba%' OR degree ILIKE '%ms%' OR degree ILIKE '%ma%' THEN 'Masters Degree'
                        WHEN degree ILIKE '%bachelor%' OR degree ILIKE '%bs%' OR degree ILIKE '%ba%' THEN 'Bachelors Degree'
                        WHEN degree ILIKE '%associate%' THEN 'Associate Degree'
                        ELSE 'Other'
                    END as education_level,
                    id
                FROM education_degrees
            )
            SELECT
                education_level,
                COUNT(DISTINCT id) as count
            FROM degree_levels
            GROUP BY education_level
            ORDER BY count DESC
        """

        async with self.db.get_connection() as conn:
            rows = await conn.fetch(query, tenant_id.value)

        values = [
            FacetValue(
                value=row['education_level'],
                count=row['count'],
                display_name=row['education_level'],
                percentage=round((row['count'] / total_profiles) * 100, 2) if total_profiles > 0 else 0.0
            )
            for row in rows
        ]

        return FacetField(
            field_name=FacetFieldName.EDUCATION_LEVEL,
            facet_type=FacetType.TERMS,
            display_name="Education Level",
            description="Highest degree obtained",
            values=values,
            searchable=False,
            multi_select=True,
            total_count=sum(v.count for v in values),
            unique_count=len(values)
        )

    async def _generate_location_facet(
        self,
        tenant_id: TenantId,
        field_name: FacetFieldName,
        total_profiles: int
    ) -> FacetField:
        """Generate location facets (country, state, city)."""
        field_map = {
            FacetFieldName.LOCATION_COUNTRY: ("location_country", "Country"),
            FacetFieldName.LOCATION_CITY: ("location_city", "City"),
        }

        db_field, display_name = field_map[field_name]

        query = f"""
            SELECT
                {db_field} as location,
                COUNT(*) as count
            FROM profiles
            WHERE tenant_id = $1
              AND status = 'active'
              AND is_deleted = false
              AND {db_field} IS NOT NULL
              AND {db_field} != ''
            GROUP BY {db_field}
            ORDER BY count DESC
            LIMIT 200
        """

        async with self.db.get_connection() as conn:
            rows = await conn.fetch(query, tenant_id.value)

        values = [
            FacetValue(
                value=row['location'],
                count=row['count'],
                display_name=row['location'],
                percentage=round((row['count'] / total_profiles) * 100, 2) if total_profiles > 0 else 0.0
            )
            for row in rows
        ]

        return FacetField(
            field_name=field_name,
            facet_type=FacetType.TERMS,
            display_name=display_name,
            description=f"Candidate {display_name.lower()}",
            values=values,
            searchable=True,
            multi_select=True,
            total_count=sum(v.count for v in values),
            unique_count=len(values)
        )

    async def _generate_companies_facet(
        self,
        tenant_id: TenantId,
        total_profiles: int
    ) -> FacetField:
        """Generate companies facet from experience data."""
        query = """
            WITH companies AS (
                SELECT
                    p.id,
                    jsonb_array_elements(p.profile_data->'experience')->>'company' as company
                FROM profiles p
                WHERE p.tenant_id = $1
                  AND p.status = 'active'
                  AND p.is_deleted = false
                  AND p.profile_data ? 'experience'
            )
            SELECT
                company,
                COUNT(DISTINCT id) as count
            FROM companies
            WHERE company IS NOT NULL
              AND company != ''
            GROUP BY company
            ORDER BY count DESC
            LIMIT 200
        """

        async with self.db.get_connection() as conn:
            rows = await conn.fetch(query, tenant_id.value)

        values = [
            FacetValue(
                value=row['company'],
                count=row['count'],
                display_name=row['company'],
                percentage=round((row['count'] / total_profiles) * 100, 2) if total_profiles > 0 else 0.0
            )
            for row in rows
        ]

        return FacetField(
            field_name=FacetFieldName.COMPANIES,
            facet_type=FacetType.TERMS,
            display_name="Previous Companies",
            description="Companies candidates have worked at",
            values=values,
            searchable=True,
            multi_select=True,
            total_count=sum(v.count for v in values),
            unique_count=len(values)
        )

    async def _generate_languages_facet(
        self,
        tenant_id: TenantId,
        total_profiles: int
    ) -> FacetField:
        """Generate languages facet."""
        query = """
            WITH languages AS (
                SELECT
                    p.id,
                    jsonb_array_elements(p.profile_data->'languages')->>'name' as language
                FROM profiles p
                WHERE p.tenant_id = $1
                  AND p.status = 'active'
                  AND p.is_deleted = false
                  AND p.profile_data ? 'languages'
            )
            SELECT
                language,
                COUNT(DISTINCT id) as count
            FROM languages
            WHERE language IS NOT NULL
              AND language != ''
            GROUP BY language
            ORDER BY count DESC
            LIMIT 50
        """

        async with self.db.get_connection() as conn:
            rows = await conn.fetch(query, tenant_id.value)

        values = [
            FacetValue(
                value=row['language'],
                count=row['count'],
                display_name=row['language'],
                percentage=round((row['count'] / total_profiles) * 100, 2) if total_profiles > 0 else 0.0
            )
            for row in rows
        ]

        return FacetField(
            field_name=FacetFieldName.LANGUAGES,
            facet_type=FacetType.TERMS,
            display_name="Languages",
            description="Languages spoken by candidates",
            values=values,
            searchable=False,
            multi_select=True,
            total_count=sum(v.count for v in values),
            unique_count=len(values)
        )

    async def get_facet_values(
        self,
        tenant_id: TenantId,
        field_name: str,
        search_query: Optional[str] = None,
        limit: int = 100
    ) -> List[FacetValue]:
        """Get values for specific facet with optional search filtering."""
        try:
            facet_field_name = FacetFieldName(field_name)
            facet_field = await self._generate_facet_field(tenant_id, facet_field_name, 1)

            if not facet_field:
                return []

            values = facet_field.values

            # Apply search filter if provided
            if search_query:
                search_lower = search_query.lower()
                values = [
                    v for v in values
                    if search_lower in v.value.lower() or search_lower in v.display_name.lower()
                ]

            return values[:limit]
        except Exception as e:
            logger.error("Failed to get facet values", field_name=field_name, error=str(e))
            return []

    async def refresh_facets(self, tenant_id: TenantId) -> FacetMetadata:
        """Force regeneration of facets."""
        return await self.generate_facets(tenant_id, force_refresh=True)

    async def get_facet_statistics(self, tenant_id: TenantId) -> Dict[str, Any]:
        """Get statistics about facet data."""
        query = """
            SELECT
                COUNT(*) as total_profiles,
                COUNT(*) FILTER (WHERE status = 'active') as active_profiles,
                COUNT(*) FILTER (WHERE array_length(normalized_skills, 1) > 0) as profiles_with_skills,
                COUNT(*) FILTER (WHERE location_country IS NOT NULL) as profiles_with_location,
                COUNT(*) FILTER (WHERE experience_level IS NOT NULL) as profiles_with_experience_level
            FROM profiles
            WHERE tenant_id = $1
              AND is_deleted = false
        """

        async with self.db.get_connection() as conn:
            row = await conn.fetchrow(query, tenant_id.value)

        return dict(row)