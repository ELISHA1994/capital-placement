"""PostgreSQL implementation of IProfileRepository using ProfileMapper."""

from __future__ import annotations

from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime, timedelta

from app.domain.entities.profile import Profile, ProfileStatus, ExperienceLevel
from app.domain.repositories.profile_repository import IProfileRepository
from app.domain.value_objects import ProfileId, TenantId, MatchScore
from app.infrastructure.persistence.mappers.profile_mapper import ProfileMapper
from app.infrastructure.persistence.models.profile_table import ProfileTable
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
            profile_table = ProfileMapper.to_table(profile)
            db_manager = adapter.db_manager

            async with db_manager.get_session() as session:
                existing_row = await session.get(ProfileTable, profile_table.id)

                if existing_row:
                    ProfileMapper.update_table_from_domain(existing_row, profile)
                else:
                    session.add(profile_table)

            return profile
            
        except Exception as e:
            raise Exception(f"Failed to save profile: {str(e)}")

    async def find_by_id(self, profile_id: ProfileId) -> Optional[Profile]:
        """Find profile by ID."""
        adapter = await self._get_adapter()
        
        try:
            db_manager = adapter.db_manager
            async with db_manager.get_session() as session:
                table_obj = await session.get(ProfileTable, profile_id.value)

                if not table_obj:
                    return None

                return ProfileMapper.to_domain(table_obj)
            
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

    # Interface method implementations
    async def get_by_id(self, profile_id: ProfileId, tenant_id: TenantId) -> Optional[Profile]:
        """Load a profile aggregate by identifier within tenant scope."""
        adapter = await self._get_adapter()

        try:
            record = await adapter.fetch_one(
                "SELECT * FROM profiles WHERE id = $1 AND tenant_id = $2",
                profile_id.value, tenant_id.value
            )

            if not record:
                return None

            profile_table = ProfileTable(**dict(record))
            return ProfileMapper.to_domain(profile_table)

        except Exception as e:
            raise Exception(f"Failed to get profile by ID: {str(e)}")

    async def get_by_email(self, email: str, tenant_id: TenantId) -> Optional[Profile]:
        """Get a profile by email within tenant scope."""
        return await self.find_by_email(tenant_id, email)

    async def list_by_tenant(
        self,
        tenant_id: TenantId,
        status: Optional[ProfileStatus] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[Profile]:
        """List profiles for a tenant with optional filtering."""
        adapter = await self._get_adapter()

        try:
            if status:
                records = await adapter.fetch_all(
                    "SELECT * FROM profiles WHERE tenant_id = $1 AND status = $2 ORDER BY created_at DESC LIMIT $3 OFFSET $4",
                    tenant_id.value, status.value, limit, offset
                )
            else:
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
            raise Exception(f"Failed to list profiles by tenant: {str(e)}")

    async def search_by_vector(
        self,
        tenant_id: TenantId,
        query_vector: List[float],
        *,
        limit: int = 20,
        threshold: float = 0.7,
    ) -> List[Tuple[Profile, MatchScore]]:
        """Run vector similarity search for candidate discovery."""
        adapter = await self._get_adapter()

        try:
            # Use pgvector cosine similarity search
            vector_str = f"[{','.join(map(str, query_vector))}]"

            records = await adapter.fetch_all(
                """
                SELECT *, 1 - (overall_embedding <=> $2::vector) as similarity
                FROM profiles
                WHERE tenant_id = $1
                AND overall_embedding IS NOT NULL
                AND 1 - (overall_embedding <=> $2::vector) >= $3
                ORDER BY similarity DESC
                LIMIT $4
                """,
                tenant_id.value, vector_str, threshold, limit
            )

            results = []
            for record in records:
                profile_data = dict(record)
                similarity = profile_data.pop('similarity', 0.0)
                profile_table = ProfileTable(**profile_data)
                profile = ProfileMapper.to_domain(profile_table)
                match_score = MatchScore(float(similarity))
                results.append((profile, match_score))

            return results

        except Exception as e:
            raise Exception(f"Failed to search by vector: {str(e)}")

    async def search_by_skills(
        self,
        tenant_id: TenantId,
        skills: List[str],
        experience_level: Optional[ExperienceLevel] = None,
        limit: int = 50
    ) -> List[Tuple[Profile, MatchScore]]:
        """Search profiles by required skills."""
        adapter = await self._get_adapter()

        try:
            # Build skill conditions
            skill_conditions = []
            params = [tenant_id.value]
            param_idx = 2

            for skill in skills:
                skill_conditions.append(f"normalized_skills @> ${param_idx}")
                params.append([skill.lower()])
                param_idx += 1

            where_clause = " AND ".join(skill_conditions)

            # Add experience level filter if provided
            if experience_level:
                where_clause += f" AND experience_level = ${param_idx}"
                params.append(experience_level.value)
                param_idx += 1

            query = f"""
                SELECT * FROM profiles
                WHERE tenant_id = $1
                AND ({where_clause})
                ORDER BY quality_score DESC NULLS LAST, created_at DESC
                LIMIT ${param_idx}
            """
            params.append(limit)

            records = await adapter.fetch_all(query, *params)

            results = []
            for record in records:
                profile_table = ProfileTable(**dict(record))
                profile = ProfileMapper.to_domain(profile_table)
                # Calculate match score based on quality score (0.0 - 1.0)
                quality = float(profile_table.quality_score or 0.5)
                match_score = MatchScore(quality)
                results.append((profile, match_score))

            return results

        except Exception as e:
            raise Exception(f"Failed to search profiles by skills: {str(e)}")

    async def delete(self, profile_id: ProfileId, tenant_id: TenantId) -> bool:
        """Delete a profile (hard delete)."""
        adapter = await self._get_adapter()

        try:
            result = await adapter.execute(
                "DELETE FROM profiles WHERE id = $1 AND tenant_id = $2",
                profile_id.value, tenant_id.value
            )

            return result and result.split()[-1] != '0'

        except Exception as e:
            raise Exception(f"Failed to delete profile: {str(e)}")

    async def count_by_tenant(
        self,
        tenant_id: TenantId,
        status: Optional[ProfileStatus] = None
    ) -> int:
        """Count profiles for a tenant."""
        adapter = await self._get_adapter()

        try:
            if status:
                record = await adapter.fetch_one(
                    "SELECT COUNT(*) as count FROM profiles WHERE tenant_id = $1 AND status = $2",
                    tenant_id.value, status.value
                )
            else:
                record = await adapter.fetch_one(
                    "SELECT COUNT(*) as count FROM profiles WHERE tenant_id = $1",
                    tenant_id.value
                )

            return record['count'] if record else 0

        except Exception as e:
            raise Exception(f"Failed to count profiles by tenant: {str(e)}")

    async def get_by_ids(
        self,
        profile_ids: List[ProfileId],
        tenant_id: TenantId
    ) -> List[Profile]:
        """Get multiple profiles by IDs."""
        adapter = await self._get_adapter()

        try:
            if not profile_ids:
                return []

            # Convert ProfileId objects to UUIDs
            id_values = [pid.value for pid in profile_ids]

            records = await adapter.fetch_all(
                "SELECT * FROM profiles WHERE id = ANY($1) AND tenant_id = $2",
                id_values, tenant_id.value
            )

            profiles = []
            for record in records:
                profile_table = ProfileTable(**dict(record))
                profiles.append(ProfileMapper.to_domain(profile_table))

            return profiles

        except Exception as e:
            raise Exception(f"Failed to get profiles by IDs: {str(e)}")

    async def update_analytics(
        self,
        profile_id: ProfileId,
        tenant_id: TenantId,
        view_increment: int = 0,
        search_appearance_increment: int = 0
    ) -> bool:
        """Update profile analytics counters."""
        adapter = await self._get_adapter()

        try:
            updates = []
            if view_increment > 0:
                updates.append(f"view_count = view_count + {view_increment}")
                updates.append("last_viewed_at = NOW()")
            if search_appearance_increment > 0:
                updates.append(f"search_appearances = search_appearances + {search_appearance_increment}")

            if not updates:
                return True  # No updates needed

            updates.append("updated_at = NOW()")
            update_clause = ", ".join(updates)

            result = await adapter.execute(
                f"UPDATE profiles SET {update_clause} WHERE id = $1 AND tenant_id = $2",
                profile_id.value, tenant_id.value
            )

            return result and result.split()[-1] != '0'

        except Exception as e:
            raise Exception(f"Failed to update analytics: {str(e)}")

    async def list_pending_processing(
        self,
        tenant_id: Optional[TenantId] = None,
        limit: int = 100
    ) -> List[Profile]:
        """List profiles pending processing."""
        adapter = await self._get_adapter()

        try:
            if tenant_id:
                records = await adapter.fetch_all(
                    """
                    SELECT * FROM profiles
                    WHERE tenant_id = $1 AND processing_status = 'pending'
                    ORDER BY created_at ASC
                    LIMIT $2
                    """,
                    tenant_id.value, limit
                )
            else:
                records = await adapter.fetch_all(
                    """
                    SELECT * FROM profiles
                    WHERE processing_status = 'pending'
                    ORDER BY created_at ASC
                    LIMIT $1
                    """,
                    limit
                )

            profiles = []
            for record in records:
                profile_table = ProfileTable(**dict(record))
                profiles.append(ProfileMapper.to_domain(profile_table))

            return profiles

        except Exception as e:
            raise Exception(f"Failed to list pending processing profiles: {str(e)}")

    async def list_for_archival(
        self,
        tenant_id: TenantId,
        days_inactive: int = 90
    ) -> List[Profile]:
        """List profiles eligible for archival."""
        adapter = await self._get_adapter()

        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days_inactive)

            records = await adapter.fetch_all(
                """
                SELECT * FROM profiles
                WHERE tenant_id = $1
                AND last_activity_at < $2
                AND status != 'archived'
                ORDER BY last_activity_at ASC
                """,
                tenant_id.value, cutoff_date
            )

            profiles = []
            for record in records:
                profile_table = ProfileTable(**dict(record))
                profiles.append(ProfileMapper.to_domain(profile_table))

            return profiles

        except Exception as e:
            raise Exception(f"Failed to list profiles for archival: {str(e)}")

    async def get_statistics(
        self,
        tenant_id: TenantId
    ) -> Dict[str, Any]:
        """Get profile statistics for a tenant."""
        adapter = await self._get_adapter()

        try:
            stats = {}

            # Total count
            total_record = await adapter.fetch_one(
                "SELECT COUNT(*) as count FROM profiles WHERE tenant_id = $1",
                tenant_id.value
            )
            stats['total_profiles'] = total_record['count'] if total_record else 0

            # Count by status
            status_records = await adapter.fetch_all(
                "SELECT status, COUNT(*) as count FROM profiles WHERE tenant_id = $1 GROUP BY status",
                tenant_id.value
            )
            stats['by_status'] = {record['status']: record['count'] for record in status_records}

            # Average quality score
            quality_record = await adapter.fetch_one(
                "SELECT AVG(quality_score) as avg_quality FROM profiles WHERE tenant_id = $1 AND quality_score IS NOT NULL",
                tenant_id.value
            )
            stats['average_quality_score'] = float(quality_record['avg_quality']) if quality_record and quality_record['avg_quality'] else 0.0

            # Profiles with embeddings
            embedding_record = await adapter.fetch_one(
                "SELECT COUNT(*) as count FROM profiles WHERE tenant_id = $1 AND overall_embedding IS NOT NULL",
                tenant_id.value
            )
            stats['profiles_with_embeddings'] = embedding_record['count'] if embedding_record else 0

            # Recent activity (last 7 days)
            recent_cutoff = datetime.utcnow() - timedelta(days=7)
            recent_record = await adapter.fetch_one(
                "SELECT COUNT(*) as count FROM profiles WHERE tenant_id = $1 AND last_activity_at >= $2",
                tenant_id.value, recent_cutoff
            )
            stats['recent_activity_count'] = recent_record['count'] if recent_record else 0

            return stats

        except Exception as e:
            raise Exception(f"Failed to get profile statistics: {str(e)}")


__all__ = ["PostgresProfileRepository"]
