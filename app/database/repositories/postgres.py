"""
SQLModel repository implementation for the CV Matching Platform.

This module provides a comprehensive repository implementation using SQLModel
with support for vector operations, transactions, and performance optimization.
Provides clean, type-safe database operations with minimal boilerplate.
"""

from typing import Any, Dict, List, Optional, Union, Type, TypeVar, Iterable
from uuid import UUID
from datetime import datetime
import structlog
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, delete, func, text
from sqlalchemy.orm import selectinload
from sqlmodel import SQLModel
import numpy as np

from app.database.error_handling import DatabaseError, QueryError, IntegrityViolationError, handle_database_errors, log_database_operation
from app.database.sqlmodel_engine import get_sqlmodel_db_manager, SQLModelDatabaseManager
from app.domain.entities.profile import Profile
from app.domain.value_objects import ProfileId
from app.infrastructure.persistence.mappers.profile_mapper import ProfileMapper
from app.infrastructure.persistence.models.auth_tables import UserTable, UserSessionTable
from app.infrastructure.persistence.models.profile_table import ProfileTable
from app.infrastructure.persistence.models.tenant_table import TenantTable, TenantConfigurationTable

logger = structlog.get_logger(__name__)

T = TypeVar('T', bound=SQLModel)


class SQLModelRepository:
    """
    Base SQLModel repository with common operations and vector support.
    Provides a unified interface for all database operations using SQLModel/SQLAlchemy.
    """
    
    def __init__(self, model_class: Type[T], db_manager: Optional[SQLModelDatabaseManager] = None):
        self.model_class = model_class
        self.db_manager = db_manager or get_sqlmodel_db_manager()
        self.logger = structlog.get_logger(f"{__name__}.{self.__class__.__name__}")

    @handle_database_errors(context={"operation": "find_by_id"})
    @log_database_operation("find_by_id")
    async def find_by_id(
        self, 
        entity_id: Union[UUID, str], 
        session: Optional[AsyncSession] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Find a record by ID using SQLModel/SQLAlchemy.
        
        Args:
            entity_id: The ID to search for
            session: Optional database session
            
        Returns:
            Dict containing the record or None if not found
        """
        async def _find():
            stmt = select(self.model_class).where(self.model_class.id == entity_id)
            result = await session.execute(stmt)
            instance = result.scalar_one_or_none()
            return instance.model_dump() if instance else None
        
        if session:
            return await _find()
        else:
            async with self.db_manager.get_session() as session:
                return await _find()

    @handle_database_errors(context={"operation": "find_by_criteria"})
    @log_database_operation("find_by_criteria")
    async def find_by_criteria(
        self,
        criteria: Dict[str, Any],
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        order_by: Optional[str] = None,
        session: Optional[AsyncSession] = None
    ) -> List[Dict[str, Any]]:
        """
        Find records by criteria using SQLModel/SQLAlchemy.
        
        Args:
            criteria: Dict of column-value pairs to filter by
            limit: Optional limit on number of results
            offset: Optional offset for pagination
            order_by: Optional order by clause
            session: Optional database session
            
        Returns:
            List of dicts containing matching records
        """
        async def _find():
            stmt = select(self.model_class)
            
            # Apply criteria filters
            for key, value in criteria.items():
                if hasattr(self.model_class, key):
                    column = getattr(self.model_class, key)
                    if value is None:
                        stmt = stmt.where(column.is_(None))
                    else:
                        stmt = stmt.where(column == value)
            
            # Apply ordering
            if order_by and hasattr(self.model_class, order_by):
                stmt = stmt.order_by(getattr(self.model_class, order_by))
            
            # Apply pagination
            if limit:
                stmt = stmt.limit(limit)
            if offset:
                stmt = stmt.offset(offset)
            
            result = await session.execute(stmt)
            return [instance.model_dump() for instance in result.scalars().all()]
        
        if session:
            return await _find()
        else:
            async with self.db_manager.get_session() as session:
                return await _find()

    @handle_database_errors(context={"operation": "create"})
    @log_database_operation("create")
    async def create(self, data: Dict[str, Any], session: Optional[AsyncSession] = None) -> Dict[str, Any]:
        """
        Create a new record using SQLModel/SQLAlchemy.
        
        Args:
            data: Dict containing the data to insert
            session: Optional database session
            
        Returns:
            Dict containing the created record
        """
        async def _create():
            # Create instance from data
            instance = self.model_class(**data)
            session.add(instance)
            await session.commit()
            await session.refresh(instance)
            return instance.model_dump()
        
        if session:
            return await _create()
        else:
            async with self.db_manager.get_session() as session:
                return await _create()

    @handle_database_errors(context={"operation": "update"})
    @log_database_operation("update")
    async def update(
        self, 
        entity_id: Union[UUID, str], 
        data: Dict[str, Any],
        session: Optional[AsyncSession] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Update a record by ID using SQLModel/SQLAlchemy.
        
        Args:
            entity_id: The ID of the record to update
            data: Dict containing the data to update
            session: Optional database session
            
        Returns:
            Dict containing the updated record or None if not found
        """
        async def _update():
            # Get existing instance
            stmt = select(self.model_class).where(self.model_class.id == entity_id)
            result = await session.execute(stmt)
            instance = result.scalar_one_or_none()
            
            if not instance:
                return None
            
            # Update fields
            for key, value in data.items():
                if hasattr(instance, key):
                    setattr(instance, key, value)
            
            # Set updated timestamp
            if hasattr(instance, 'updated_at'):
                setattr(instance, 'updated_at', datetime.utcnow())
            
            session.add(instance)
            await session.commit()
            await session.refresh(instance)
            return instance.model_dump()
        
        if session:
            return await _update()
        else:
            async with self.db_manager.get_session() as session:
                return await _update()

    @handle_database_errors(context={"operation": "delete"})
    @log_database_operation("delete")
    async def delete(self, entity_id: Union[UUID, str], session: Optional[AsyncSession] = None) -> bool:
        """
        Delete a record by ID using SQLModel/SQLAlchemy.
        
        Args:
            entity_id: The ID of the record to delete
            session: Optional database session
            
        Returns:
            True if record was deleted, False if not found
        """
        async def _delete():
            # Get existing instance
            stmt = select(self.model_class).where(self.model_class.id == entity_id)
            result = await session.execute(stmt)
            instance = result.scalar_one_or_none()
            
            if not instance:
                return False
            
            await session.delete(instance)
            await session.commit()
            return True
        
        if session:
            return await _delete()
        else:
            async with self.db_manager.get_session() as session:
                return await _delete()

    @handle_database_errors(context={"operation": "count"})
    async def count(self, criteria: Optional[Dict[str, Any]] = None, session: Optional[AsyncSession] = None) -> int:
        """
        Count records matching criteria using SQLModel/SQLAlchemy.
        
        Args:
            criteria: Optional dict of column-value pairs to filter by
            session: Optional database session
            
        Returns:
            Count of matching records
        """
        async def _count():
            stmt = select(func.count()).select_from(self.model_class)
            
            # Apply criteria filters
            if criteria:
                for key, value in criteria.items():
                    if hasattr(self.model_class, key):
                        column = getattr(self.model_class, key)
                        if value is None:
                            stmt = stmt.where(column.is_(None))
                        else:
                            stmt = stmt.where(column == value)
            
            result = await session.execute(stmt)
            return result.one()
        
        if session:
            return await _count()
        else:
            async with self.db_manager.get_session() as session:
                return await _count()

    @handle_database_errors(context={"operation": "bulk_create"})
    @log_database_operation("bulk_create")
    async def bulk_create(self, data_list: List[Dict[str, Any]], session: Optional[AsyncSession] = None) -> List[Dict[str, Any]]:
        """
        Create multiple records in a single transaction using SQLModel/SQLAlchemy.
        
        Args:
            data_list: List of dicts containing the data to insert
            session: Optional database session
            
        Returns:
            List of dicts containing the created records
        """
        if not data_list:
            return []
        
        async def _bulk_create():
            results = []
            for data in data_list:
                instance = self.model_class(**data)
                session.add(instance)
                results.append(instance)
            
            await session.commit()
            
            # Refresh all instances and return as dicts
            refreshed_results = []
            for instance in results:
                await session.refresh(instance)
                refreshed_results.append(instance.model_dump())
            
            return refreshed_results
        
        if session:
            return await _bulk_create()
        else:
            async with self.db_manager.get_session() as session:
                return await _bulk_create()

    # Standard interface methods
    async def get_by_id(self, entity_id: Union[UUID, str], session: Optional[AsyncSession] = None) -> Optional[Dict[str, Any]]:
        """Get entity by ID (standard interface method)."""
        return await self.find_by_id(entity_id, session)
    
    async def list(
        self,
        offset: int = 0,
        limit: int = 20,
        filters: Optional[Dict[str, Any]] = None,
        session: Optional[AsyncSession] = None
    ) -> List[Dict[str, Any]]:
        """List entities with pagination (standard interface method)."""
        criteria = filters or {}
        return await self.find_by_criteria(
            criteria=criteria,
            limit=limit,
            offset=offset,
            session=session
        )


class VectorRepository(SQLModelRepository):
    """
    Extended repository with vector similarity search capabilities using SQLModel/SQLAlchemy.
    """

    @handle_database_errors(context={"operation": "vector_similarity_search"})
    @log_database_operation("vector_similarity_search")
    async def vector_similarity_search(
        self,
        query_vector: Union[List[float], np.ndarray],
        vector_column: str,
        limit: int = 10,
        threshold: float = 0.7,
        additional_criteria: Optional[Dict[str, Any]] = None,
        distance_metric: str = "cosine",
        session: Optional[AsyncSession] = None
    ) -> List[Dict[str, Any]]:
        """
        Perform vector similarity search using SQLModel/SQLAlchemy with pgvector.
        
        Args:
            query_vector: The query vector to search for similar vectors
            vector_column: Name of the vector column to search
            limit: Maximum number of results to return
            threshold: Minimum similarity threshold (0-1)
            additional_criteria: Optional additional WHERE conditions
            distance_metric: Distance metric to use ('cosine', 'euclidean', 'inner_product')
            session: Optional database session
            
        Returns:
            List of dicts containing matching records with similarity scores
        """
        async def _search():
            # Convert vector to proper format
            if isinstance(query_vector, np.ndarray):
                query_vector_list = query_vector.tolist()
            else:
                query_vector_list = query_vector
            
            vector_str = f"[{','.join(map(str, query_vector_list))}]"
            
            # Choose distance operator
            distance_ops = {
                "cosine": "<=>",
                "euclidean": "<->", 
                "inner_product": "<#>"
            }
            
            distance_op = distance_ops.get(distance_metric, "<=>")
            
            # Build similarity expression  
            similarity_expr = f"1 - ({vector_column} {distance_op} '{vector_str}'::vector)"
            table_name = self.model_class.__tablename__
            
            # Build query with raw SQL for vector operations
            query = f"""
                SELECT *, {similarity_expr} as similarity_score
                FROM {table_name}
                WHERE {vector_column} IS NOT NULL
            """
            
            # Add additional criteria
            if additional_criteria:
                for key, value in additional_criteria.items():
                    if value is None:
                        query += f" AND {key} IS NULL"
                    else:
                        query += f" AND {key} = '{value}'"
            
            # Add similarity threshold
            if threshold > 0:
                query += f" AND {similarity_expr} >= {threshold}"
            
            # Order by similarity and limit
            query += f" ORDER BY {vector_column} {distance_op} '{vector_str}'::vector LIMIT {limit}"
            
            result = await session.execute(text(query))
            rows = result.all()
            
            # Convert rows to dicts
            results = []
            for row in rows:
                row_dict = dict(row._mapping) if hasattr(row, '_mapping') else dict(row)
                results.append(row_dict)
            
            return results
        
        if session:
            return await _search()
        else:
            async with self.db_manager.get_session() as session:
                return await _search()

    @handle_database_errors(context={"operation": "update_vector_embedding"})
    @log_database_operation("update_vector_embedding")
    async def update_vector_embedding(
        self,
        entity_id: Union[UUID, str],
        vector_column: str,
        embedding: Union[List[float], np.ndarray],
        session: Optional[AsyncSession] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Update vector embedding for a record using SQLModel/SQLAlchemy.
        
        Args:
            entity_id: ID of the record to update
            vector_column: Name of the vector column to update
            embedding: The embedding vector to store
            session: Optional database session
            
        Returns:
            Updated record or None if not found
        """
        async def _update_embedding():
            if isinstance(embedding, np.ndarray):
                embedding_list = embedding.tolist()
            else:
                embedding_list = embedding
            
            # Update using the standard update method
            return await self.update(entity_id, {vector_column: embedding_list}, session)
        
        if session:
            return await _update_embedding()
        else:
            async with self.db_manager.get_session() as session:
                return await _update_embedding()

    @handle_database_errors(context={"operation": "batch_vector_search"})
    @log_database_operation("batch_vector_search")
    async def batch_vector_similarity_search(
        self,
        query_vectors: List[Union[List[float], np.ndarray]],
        vector_column: str,
        limit_per_query: int = 10,
        threshold: float = 0.7,
        distance_metric: str = "cosine",
        session: Optional[AsyncSession] = None
    ) -> List[List[Dict[str, Any]]]:
        """
        Perform batch vector similarity searches using SQLModel/SQLAlchemy.
        
        Args:
            query_vectors: List of query vectors
            vector_column: Name of the vector column to search
            limit_per_query: Maximum results per query
            threshold: Minimum similarity threshold
            distance_metric: Distance metric to use
            session: Optional database session
            
        Returns:
            List of result lists (one per query vector)
        """
        async def _batch_search():
            results = []
            for query_vector in query_vectors:
                query_results = await self.vector_similarity_search(
                    query_vector=query_vector,
                    vector_column=vector_column,
                    limit=limit_per_query,
                    threshold=threshold,
                    distance_metric=distance_metric,
                    session=session
                )
                results.append(query_results)
            return results
        
        if session:
            return await _batch_search()
        else:
            async with self.db_manager.get_session() as session:
                return await _batch_search()


class TenantRepository(SQLModelRepository):
    """Repository for tenant management using SQLModel."""
    
    def __init__(self):
        super().__init__(TenantTable)

    # SQLModel handles JSONB serialization/deserialization automatically

    @handle_database_errors(context={"operation": "find_by_slug"})
    async def find_by_slug(self, slug: str, session: Optional[AsyncSession] = None) -> Optional[Dict[str, Any]]:
        """Find tenant by slug using SQLModel."""
        async def _find():
            stmt = select(TenantTable).where(TenantTable.slug == slug)
            result = await session.execute(stmt)
            instance = result.scalar_one_or_none()
            return instance.model_dump() if instance else None
        
        if session:
            return await _find()
        else:
            async with self.db_manager.get_session() as session:
                return await _find()
    
    @handle_database_errors(context={"operation": "get_by_id"})
    async def get(self, tenant_id: Union[UUID, str], session: Optional[AsyncSession] = None) -> Optional[Dict[str, Any]]:
        """Get tenant by ID (alias for find_by_id)."""
        return await self.find_by_id(tenant_id, session)
    
    @handle_database_errors(context={"operation": "list_all"})
    async def list_all(self, session: Optional[AsyncSession] = None) -> List[Dict[str, Any]]:
        """List all tenants using SQLModel."""
        return await self.find_by_criteria({}, session=session)
    
    @handle_database_errors(context={"operation": "check_slug_availability"})
    async def check_slug_availability(self, slug: str, exclude_id: Optional[str] = None, session: Optional[AsyncSession] = None) -> bool:
        """Check if tenant slug is available using SQLModel."""
        async def _check():
            # Check both slug and name fields for uniqueness since tenant names must be unique
            stmt = select(func.count()).select_from(TenantTable).where(
                (TenantTable.slug == slug) | (TenantTable.name == slug)
            )
            
            if exclude_id:
                stmt = stmt.where(TenantTable.id != exclude_id)
            
            result = await session.execute(stmt)
            count = result.scalar_one()
            return count == 0
        
        if session:
            return await _check()
        else:
            async with self.db_manager.get_session() as session:
                return await _check()


class UserRepository(SQLModelRepository):
    """
    User repository using SQLModel with automatic schema handling.
    SQLModel handles all serialization, type validation, and mapping automatically.
    """
    
    def __init__(self):
        super().__init__(UserTable)

    async def get_by_email(self, email: str, tenant_id: Union[UUID, str], session: Optional[AsyncSession] = None) -> Optional[Dict[str, Any]]:
        """Get user by email and tenant_id using SQLModel."""
        try:
            results = await self.find_by_criteria({
                'email': email.lower(),
                'tenant_id': str(tenant_id)
            }, limit=1, session=session)
            return results[0] if results else None
        except Exception as e:
            logger.error("Failed to get user by email", 
                        email=email, tenant_id=tenant_id, error=str(e))
            return None

    # SQLModel handles all schema mapping, serialization, and validation automatically

    async def get_by_tenant(self, tenant_id: str, session: Optional[AsyncSession] = None) -> List[Dict[str, Any]]:
        """Get all users for a specific tenant using SQLModel."""
        return await self.find_by_criteria({"tenant_id": tenant_id}, session=session)
    
    async def get_by_email_and_tenant(self, email: str, tenant_id: str, session: Optional[AsyncSession] = None) -> Optional[Dict[str, Any]]:
        """Get user by email and tenant combination using SQLModel."""
        results = await self.find_by_criteria({
            "email": email.lower(), 
            "tenant_id": tenant_id
        }, limit=1, session=session)
        return results[0] if results else None
    
    async def update_last_login(self, user_id: Union[UUID, str], session: Optional[AsyncSession] = None) -> None:
        """Update user's last login timestamp using SQLModel."""
        from datetime import datetime, timezone
        await self.update(user_id, {"last_login_at": datetime.now(timezone.utc)}, session=session)


class UserSessionRepository(SQLModelRepository):
    """Repository for user sessions."""

    def __init__(self):
        super().__init__(UserSessionTable)

    async def list_by_user(
        self,
        user_id: Union[UUID, str],
        session: Optional[AsyncSession] = None
    ) -> List[Dict[str, Any]]:
        user_id_value = UUID(str(user_id)) if not isinstance(user_id, UUID) else user_id
        return await self.find_by_criteria({"user_id": user_id_value}, session=session)

    async def delete_by_user(
        self,
        user_id: Union[UUID, str],
        session: Optional[AsyncSession] = None
    ) -> int:
        user_id_value = UUID(str(user_id)) if not isinstance(user_id, UUID) else user_id

        async def _delete():
            stmt = delete(self.model_class).where(self.model_class.user_id == user_id_value)
            result = await session.execute(stmt)
            await session.commit()
            return result.rowcount or 0

        if session:
            return await _delete()
        else:
            async with self.db_manager.get_session() as session:
                return await _delete()

class ProfileRepository(VectorRepository):
    """Repository for profile management with vector search capabilities."""
    
    def __init__(self):
        super().__init__(ProfileTable)
        self._mapper = ProfileMapper()

    async def _load_profile(
        self,
        profile_id: Union[UUID, str],
        *,
        session: AsyncSession,
    ) -> Optional[Profile]:
        stmt = select(ProfileTable).where(ProfileTable.id == profile_id)
        result = await session.execute(stmt)
        instance = result.scalar_one_or_none()
        if not instance:
            return None
        return self._mapper.to_domain(instance)

    async def get_profile(self, profile_id: ProfileId) -> Optional[Profile]:
        """Load a profile aggregate by identifier."""

        async with self.db_manager.get_session() as session:
            return await self._load_profile(profile_id.value, session=session)

    async def save_profile(self, profile: Profile) -> Profile:
        """Persist a profile aggregate using SQLModel infrastructure."""

        async with self.db_manager.get_session() as session:
            stmt = select(ProfileTable).where(ProfileTable.id == profile.id.value)
            result = await session.execute(stmt)
            instance = result.scalar_one_or_none()

            model = self._mapper.to_model(profile, existing=instance)
            session.add(model)
            await session.commit()
            await session.refresh(model)
            return self._mapper.to_domain(model)

    async def _rows_to_domain(
        self,
        rows: Iterable[Dict[str, Any]],
        *,
        session: AsyncSession,
    ) -> List[Profile]:
        ids: List[UUID] = []
        for row in rows:
            raw_id = row.get("id") if isinstance(row, dict) else None
            if raw_id is None:
                continue
            ids.append(raw_id if isinstance(raw_id, UUID) else UUID(str(raw_id)))

        if not ids:
            return []

        stmt = select(ProfileTable).where(ProfileTable.id.in_(ids))
        result = await session.execute(stmt)
        instances = result.scalars().all()
        by_id = {instance.id: instance for instance in instances}

        ordered: List[Profile] = []
        for identifier in ids:
            instance = by_id.get(identifier)
            if instance:
                ordered.append(self._mapper.to_domain(instance))
        return ordered
    
    async def search_profiles_by_vector(
        self,
        query_vector: Union[List[float], np.ndarray],
        tenant_id: UUID,
        vector_column: str = "overall_embedding",
        limit: int = 20,
        threshold: float = 0.7,
        filters: Optional[Dict[str, Any]] = None,
        session: Optional[AsyncSession] = None,
        *,
        as_domain: bool = False,
    ) -> Union[List[Dict[str, Any]], List[Profile]]:
        """Search profiles using vector similarity with optional domain mapping."""
        criteria = {"tenant_id": tenant_id}
        if filters:
            criteria.update(filters)

        async def _search(active_session: AsyncSession):
            rows = await self.vector_similarity_search(
                query_vector=query_vector,
                vector_column=vector_column,
                limit=limit,
                threshold=threshold,
                additional_criteria=criteria,
                session=active_session,
            )
            if not as_domain:
                return rows
            return await self._rows_to_domain(rows, session=active_session)

        if session:
            return await _search(session)

        async with self.db_manager.get_session() as new_session:
            return await _search(new_session)


class JobRepository(SQLModelRepository):
    """
    Job repository - placeholder implementation.
    To be completed when job models are available.
    """
    
    def __init__(self):
        # Using ProfileTable as a placeholder until JobTable is created
        super().__init__(ProfileTable)


class CandidateRepository(SQLModelRepository):
    """
    Candidate repository - placeholder implementation.
    To be completed when candidate models are available.
    """
    
    def __init__(self):
        # Using ProfileTable as a placeholder until CandidateTable is created
        super().__init__(ProfileTable)


class MatchRepository(SQLModelRepository):
    """
    Match repository - placeholder implementation.
    To be completed when match models are available.
    """
    
    def __init__(self):
        # Using ProfileTable as placeholder until MatchTable is created
        super().__init__(ProfileTable)
