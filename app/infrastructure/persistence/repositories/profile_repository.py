"""PostgreSQL implementation of IProfileRepository using ProfileMapper."""

from __future__ import annotations

from typing import List, Optional, Dict, Any
from uuid import UUID

from app.domain.entities.profile import Profile
from app.domain.repositories.profile_repository import IProfileRepository
from app.domain.value_objects import ProfileId, TenantId
from app.infrastructure.persistence.mappers.profile_mapper import ProfileMapper
from app.models.profile import ProfileTable
from app.infrastructure.providers.postgres_provider import get_postgres_adapter


class PostgresProfileRepository(IProfileRepository):
    """PostgreSQL adapter implementation of IProfileRepository."""

    def __init__(self):
        self._adapter = None

    async def _get_adapter(self):
        """Get database adapter (lazy initialization)."""
        if self._adapter is None:
            self._adapter = await get_postgres_adapter()
        return self._adapter

    async def save(self, profile: Profile) -> Profile:
        """Save profile to database and return updated domain entity."""
        adapter = await self._get_adapter()
        
        try:
            # Check if profile exists
            existing = await self.find_by_id(profile.id)
            
            if existing:
                # Update existing profile
                profile_table = ProfileMapper.to_persistence(profile)
                
                # Use UPDATE query
                await adapter.execute(
                    """
                    UPDATE profiles 
                    SET tenant_id = $2, status = $3, experience_level = $4, 
                        profile_data = $5, searchable_text = $6, keywords = $7,
                        normalized_skills = $8, name = $9, email = $10, phone = $11,
                        location_city = $12, location_state = $13, location_country = $14,
                        overall_embedding = $15, skills_embedding = $16, 
                        experience_embedding = $17, summary_embedding = $18,
                        processing_status = $19, processing_metadata = $20, quality_score = $21,
                        privacy_settings = $22, consent_given = $23, consent_date = $24,
                        view_count = $25, search_appearances = $26, last_viewed_at = $27,
                        last_activity_at = $28, updated_at = $29
                    WHERE id = $1
                    """,
                    profile_table.id, profile_table.tenant_id, profile_table.status,
                    profile_table.experience_level, profile_table.profile_data,
                    profile_table.searchable_text, profile_table.keywords,
                    profile_table.normalized_skills, profile_table.name,
                    profile_table.email, profile_table.phone, profile_table.location_city,
                    profile_table.location_state, profile_table.location_country,
                    profile_table.overall_embedding, profile_table.skills_embedding,
                    profile_table.experience_embedding, profile_table.summary_embedding,
                    profile_table.processing_status, profile_table.processing_metadata,
                    profile_table.quality_score, profile_table.privacy_settings,
                    profile_table.consent_given, profile_table.consent_date,
                    profile_table.view_count, profile_table.search_appearances,
                    profile_table.last_viewed_at, profile_table.last_activity_at,
                    profile_table.updated_at
                )
            else:
                # Insert new profile
                profile_table = ProfileMapper.to_persistence(profile)
                
                await adapter.execute(
                    """
                    INSERT INTO profiles (
                        id, tenant_id, status, experience_level, profile_data, 
                        searchable_text, keywords, normalized_skills, name, email, phone,
                        location_city, location_state, location_country, overall_embedding,
                        skills_embedding, experience_embedding, summary_embedding,
                        processing_status, processing_metadata, quality_score,
                        privacy_settings, consent_given, consent_date, view_count,
                        search_appearances, last_viewed_at, last_activity_at,
                        created_at, updated_at
                    ) VALUES (
                        $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14,
                        $15, $16, $17, $18, $19, $20, $21, $22, $23, $24, $25, $26,
                        $27, $28, $29, $30
                    )
                    """,
                    profile_table.id, profile_table.tenant_id, profile_table.status,
                    profile_table.experience_level, profile_table.profile_data,
                    profile_table.searchable_text, profile_table.keywords,
                    profile_table.normalized_skills, profile_table.name,
                    profile_table.email, profile_table.phone, profile_table.location_city,
                    profile_table.location_state, profile_table.location_country,
                    profile_table.overall_embedding, profile_table.skills_embedding,
                    profile_table.experience_embedding, profile_table.summary_embedding,
                    profile_table.processing_status, profile_table.processing_metadata,
                    profile_table.quality_score, profile_table.privacy_settings,
                    profile_table.consent_given, profile_table.consent_date,
                    profile_table.view_count, profile_table.search_appearances,
                    profile_table.last_viewed_at, profile_table.last_activity_at,
                    profile_table.created_at, profile_table.updated_at
                )
            
            # Return the saved profile (could refetch from DB to ensure consistency)
            return profile
            
        except Exception as e:
            raise Exception(f"Failed to save profile: {str(e)}")

    async def find_by_id(self, profile_id: ProfileId) -> Optional[Profile]:
        """Find profile by ID."""
        adapter = await self._get_adapter()
        
        try:
            record = await adapter.fetch_one(
                "SELECT * FROM profiles WHERE id = $1",
                profile_id.value
            )
            
            if not record:
                return None
            
            # Convert record to ProfileTable and then to domain entity
            profile_table = ProfileTable(**dict(record))
            return ProfileMapper.to_domain(profile_table)
            
        except Exception as e:
            raise Exception(f"Failed to find profile by ID: {str(e)}")

    async def find_by_tenant_id(self, tenant_id: TenantId, limit: int = 100, offset: int = 0) -> List[Profile]:
        """Find profiles by tenant ID with pagination."""
        adapter = await self._get_adapter()
        
        try:
            records = await adapter.fetch_all(
                "SELECT * FROM profiles WHERE tenant_id = $1 ORDER BY created_at DESC LIMIT $2 OFFSET $3",
                tenant_id.value, limit, offset
            )
            
            profiles = []
            for record in records:
                profile_table = ProfileTable(**dict(record))
                profiles.append(ProfileMapper.to_domain(profile_table))
            
            return profiles
            
        except Exception as e:
            raise Exception(f"Failed to find profiles by tenant ID: {str(e)}")

    async def search_by_skills(self, tenant_id: TenantId, skills: List[str], limit: int = 50) -> List[Profile]:
        """Search profiles by skills within a tenant."""
        adapter = await self._get_adapter()
        
        try:
            # Simple skill search using JSONB contains
            skill_conditions = []
            params = [tenant_id.value]
            
            for i, skill in enumerate(skills, 2):
                skill_conditions.append(f"normalized_skills @> $${i}")
                params.append([skill.lower()])
            
            where_clause = " AND ".join(skill_conditions)
            
            query = f"""
                SELECT * FROM profiles 
                WHERE tenant_id = $1 
                AND ({where_clause})
                ORDER BY quality_score DESC NULLS LAST, created_at DESC 
                LIMIT ${len(params) + 1}
            """
            params.append(limit)
            
            records = await adapter.fetch_all(query, *params)
            
            profiles = []
            for record in records:
                profile_table = ProfileTable(**dict(record))
                profiles.append(ProfileMapper.to_domain(profile_table))
            
            return profiles
            
        except Exception as e:
            raise Exception(f"Failed to search profiles by skills: {str(e)}")

    async def search_by_text(self, tenant_id: TenantId, query_text: str, limit: int = 50) -> List[Profile]:
        """Search profiles by text content within a tenant."""
        adapter = await self._get_adapter()
        
        try:
            # Full-text search using PostgreSQL's to_tsvector
            records = await adapter.fetch_all(
                """
                SELECT *, ts_rank(to_tsvector('english', searchable_text), plainto_tsquery('english', $2)) as rank
                FROM profiles 
                WHERE tenant_id = $1 
                AND to_tsvector('english', searchable_text) @@ plainto_tsquery('english', $2)
                ORDER BY rank DESC, quality_score DESC NULLS LAST
                LIMIT $3
                """,
                tenant_id.value, query_text, limit
            )
            
            profiles = []
            for record in records:
                # Remove the rank field before creating ProfileTable
                profile_data = dict(record)
                profile_data.pop('rank', None)
                profile_table = ProfileTable(**profile_data)
                profiles.append(ProfileMapper.to_domain(profile_table))
            
            return profiles
            
        except Exception as e:
            raise Exception(f"Failed to search profiles by text: {str(e)}")

    async def delete_by_id(self, profile_id: ProfileId) -> bool:
        """Delete profile by ID."""
        adapter = await self._get_adapter()
        
        try:
            result = await adapter.execute(
                "DELETE FROM profiles WHERE id = $1",
                profile_id.value
            )
            
            # Check if any rows were affected
            return result and result.split()[-1] != '0'
            
        except Exception as e:
            raise Exception(f"Failed to delete profile: {str(e)}")

    async def count_by_tenant_id(self, tenant_id: TenantId) -> int:
        """Count profiles by tenant ID."""
        adapter = await self._get_adapter()
        
        try:
            record = await adapter.fetch_one(
                "SELECT COUNT(*) as count FROM profiles WHERE tenant_id = $1",
                tenant_id.value
            )
            
            return record['count'] if record else 0
            
        except Exception as e:
            raise Exception(f"Failed to count profiles: {str(e)}")

    async def find_by_email(self, tenant_id: TenantId, email: str) -> Optional[Profile]:
        """Find profile by email within a tenant."""
        adapter = await self._get_adapter()
        
        try:
            record = await adapter.fetch_one(
                "SELECT * FROM profiles WHERE tenant_id = $1 AND email = $2",
                tenant_id.value, email
            )
            
            if not record:
                return None
            
            profile_table = ProfileTable(**dict(record))
            return ProfileMapper.to_domain(profile_table)
            
        except Exception as e:
            raise Exception(f"Failed to find profile by email: {str(e)}")

    async def update_view_count(self, profile_id: ProfileId) -> None:
        """Increment profile view count."""
        adapter = await self._get_adapter()
        
        try:
            await adapter.execute(
                """
                UPDATE profiles 
                SET view_count = view_count + 1, 
                    last_viewed_at = NOW(),
                    updated_at = NOW()
                WHERE id = $1
                """,
                profile_id.value
            )
            
        except Exception as e:
            raise Exception(f"Failed to update view count: {str(e)}")

    async def update_search_appearances(self, profile_id: ProfileId) -> None:
        """Increment profile search appearances count."""
        adapter = await self._get_adapter()
        
        try:
            await adapter.execute(
                """
                UPDATE profiles 
                SET search_appearances = search_appearances + 1,
                    updated_at = NOW()
                WHERE id = $1
                """,
                profile_id.value
            )
            
        except Exception as e:
            raise Exception(f"Failed to update search appearances: {str(e)}")


__all__ = ["PostgresProfileRepository"]