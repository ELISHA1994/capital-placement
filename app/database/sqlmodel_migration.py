"""
SQLModel Auto-Migration System for Development.

This module provides automatic table creation and updates for development environments
while preserving explicit migration control for production environments.
"""

from typing import List, Dict, Any, Optional
from pathlib import Path
import structlog
from sqlalchemy import text, inspect
from sqlalchemy.ext.asyncio import AsyncEngine
from sqlmodel import SQLModel

from .error_handling import (
    DatabaseError,
    handle_database_errors,
    log_database_operation,
)
from .sqlmodel_engine import get_sqlmodel_db_manager
from app.core.environment import get_current_environment, Environment
from app.models.auth import UserTable
from app.models.profile import ProfileTable  
from app.models.tenant_models import TenantTable, TenantConfigurationTable

logger = structlog.get_logger(__name__)


class SQLModelAutoMigration:
    """
    Handles automatic SQLModel table creation and updates for development.
    Uses SQLModel metadata for automatic schema generation.
    """
    
    def __init__(self, engine: AsyncEngine):
        self.engine = engine
        self.logger = structlog.get_logger(f"{__name__}.{self.__class__.__name__}")
        
        # Register all SQLModel tables here
        self._registered_tables = [
            TenantTable,  # Core tenant table (referenced by foreign keys)
            UserTable,
            ProfileTable, 
            TenantConfigurationTable,
        ]
    
    @property
    def should_auto_migrate(self) -> bool:
        """Check if auto-migration should be enabled based on environment."""
        env = get_current_environment()
        return env in [Environment.LOCAL, Environment.DEVELOPMENT]
    
    @handle_database_errors(context={"operation": "check_table_exists"})
    async def _table_exists(self, table_name: str) -> bool:
        """Check if a table exists in the database."""
        async with self.engine.begin() as conn:
            result = await conn.execute(text("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = :table_name
                )
            """), {"table_name": table_name})
            return result.scalar()
    
    @handle_database_errors(context={"operation": "get_table_columns"})
    async def _get_table_columns(self, table_name: str) -> Dict[str, Any]:
        """Get existing table column information."""
        async with self.engine.begin() as conn:
            inspector = inspect(conn.sync_connection)
            try:
                columns = inspector.get_columns(table_name)
                return {col['name']: col for col in columns}
            except Exception:
                return {}
    
    @handle_database_errors(context={"operation": "create_extensions"})
    @log_database_operation("create_extensions")
    async def _ensure_extensions(self) -> None:
        """Ensure required PostgreSQL extensions are installed."""
        async with self.engine.begin() as conn:
            # Create pgvector extension for vector operations
            await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            
            # Create uuid-ossp extension for UUID generation
            await conn.execute(text("CREATE EXTENSION IF NOT EXISTS \"uuid-ossp\""))
            
            # Create pg_trgm for text search optimizations
            await conn.execute(text("CREATE EXTENSION IF NOT EXISTS pg_trgm"))
            
            self.logger.info("Database extensions ensured")
    
    @handle_database_errors(context={"operation": "create_tables"})
    @log_database_operation("create_all_tables")
    async def create_all_tables(self) -> None:
        """
        Create all SQLModel tables using metadata.
        Only runs in development environments.
        """
        if not self.should_auto_migrate:
            self.logger.info(
                "Auto-migration disabled for production environment",
                environment=get_current_environment().value
            )
            return
        
        self.logger.info("Starting SQLModel auto-migration")
        
        # Ensure required extensions
        await self._ensure_extensions()
        
        # Create all tables from SQLModel metadata
        async with self.engine.begin() as conn:
            await conn.run_sync(SQLModel.metadata.create_all)
        
        self.logger.info(
            "SQLModel tables created successfully",
            table_count=len(SQLModel.metadata.tables)
        )
    
    @handle_database_errors(context={"operation": "drop_tables"})
    @log_database_operation("drop_all_tables")
    async def drop_all_tables(self) -> None:
        """
        Drop all SQLModel tables.
        Only for development/testing - NEVER use in production.
        """
        if not self.should_auto_migrate:
            raise DatabaseError("Table dropping disabled in production environment")
        
        self.logger.warning("Dropping all SQLModel tables - DESTRUCTIVE OPERATION")
        
        async with self.engine.begin() as conn:
            await conn.run_sync(SQLModel.metadata.drop_all)
        
        self.logger.warning("All SQLModel tables dropped")
    
    @handle_database_errors(context={"operation": "check_schema_drift"})
    async def detect_schema_changes(self) -> List[str]:
        """
        Detect schema changes between SQLModel definitions and database.
        Returns list of detected differences.
        """
        changes = []
        
        for table_class in self._registered_tables:
            table_name = table_class.__tablename__
            
            # Check if table exists
            exists = await self._table_exists(table_name)
            if not exists:
                changes.append(f"Table '{table_name}' missing - needs creation")
                continue
            
            # Get current columns
            db_columns = await self._get_table_columns(table_name)
            
            # Get SQLModel columns
            model_columns = {
                name: column for name, column in table_class.__table__.columns.items()
            }
            
            # Check for missing columns
            for col_name in model_columns:
                if col_name not in db_columns:
                    changes.append(f"Column '{table_name}.{col_name}' missing")
            
            # Check for extra columns
            for col_name in db_columns:
                if col_name not in model_columns:
                    changes.append(f"Column '{table_name}.{col_name}' not in model")
        
        return changes
    
    @handle_database_errors(context={"operation": "auto_sync_schema"})
    @log_database_operation("auto_sync_schema")
    async def sync_schema_changes(self) -> Dict[str, Any]:
        """
        Automatically synchronize schema changes in development.
        Returns summary of changes made.
        """
        if not self.should_auto_migrate:
            return {"status": "disabled", "reason": "production environment"}
        
        changes = await self.detect_schema_changes()
        
        if not changes:
            self.logger.info("No schema changes detected")
            return {"status": "no_changes", "changes": []}
        
        self.logger.info("Schema drift detected", changes=changes)
        
        # In development, we can recreate tables safely
        # For more sophisticated sync, we'd need Alembic integration
        await self.create_all_tables()
        
        return {
            "status": "synced",
            "changes": changes,
            "action": "tables_recreated"
        }
    
    @handle_database_errors(context={"operation": "get_table_info"})
    async def get_database_info(self) -> Dict[str, Any]:
        """Get information about current database schema."""
        async with self.engine.begin() as conn:
            # Get table count
            result = await conn.execute(text("""
                SELECT COUNT(*) FROM information_schema.tables 
                WHERE table_schema = 'public' AND table_type = 'BASE TABLE'
            """))
            table_count = result.scalar()
            
            # Get extension info
            result = await conn.execute(text("""
                SELECT extname FROM pg_extension 
                WHERE extname IN ('vector', 'uuid-ossp', 'pg_trgm')
            """))
            extensions = [row[0] for row in result.fetchall()]
            
            # Get SQLModel metadata info
            sqlmodel_tables = list(SQLModel.metadata.tables.keys())
            
            return {
                "database_tables": table_count,
                "sqlmodel_tables": len(sqlmodel_tables),
                "sqlmodel_table_names": sqlmodel_tables,
                "extensions": extensions,
                "auto_migration_enabled": self.should_auto_migrate,
                "environment": get_current_environment().value
            }


# Global auto-migration instance
_auto_migration: Optional[SQLModelAutoMigration] = None


async def get_auto_migration() -> SQLModelAutoMigration:
    """Get or create the auto-migration instance."""
    global _auto_migration
    
    if _auto_migration is None:
        db_manager = get_sqlmodel_db_manager()
        if not db_manager.is_initialized:
            await db_manager.initialize()
        _auto_migration = SQLModelAutoMigration(db_manager.engine)
    
    return _auto_migration


@handle_database_errors(context={"operation": "initialize_sqlmodel_tables"})
@log_database_operation("initialize_sqlmodel_tables")
async def initialize_sqlmodel_tables() -> Dict[str, Any]:
    """
    Initialize SQLModel tables based on environment.
    
    - Development: Auto-create/update tables
    - Production: Require explicit migrations
    
    Returns:
        Dict containing initialization status and information
    """
    auto_migration = await get_auto_migration()
    
    if auto_migration.should_auto_migrate:
        # Development: Auto-create tables
        await auto_migration.create_all_tables()
        sync_result = await auto_migration.sync_schema_changes()
        
        info = await auto_migration.get_database_info()
        
        return {
            "status": "auto_migrated",
            "sync_result": sync_result,
            "database_info": info
        }
    else:
        # Production: Just return info, don't auto-create
        info = await auto_migration.get_database_info()
        
        return {
            "status": "production_mode",
            "message": "Use explicit migrations in production",
            "database_info": info
        }


async def reset_development_database() -> Dict[str, Any]:
    """
    Reset database in development environment.
    DESTRUCTIVE - Only works in development.
    """
    auto_migration = await get_auto_migration()
    
    if not auto_migration.should_auto_migrate:
        raise DatabaseError("Database reset only allowed in development")
    
    logger.warning("Resetting development database - ALL DATA WILL BE LOST")
    
    # Drop and recreate all tables
    await auto_migration.drop_all_tables()
    await auto_migration.create_all_tables()
    
    info = await auto_migration.get_database_info()
    
    return {
        "status": "reset_complete",
        "database_info": info,
        "warning": "All data was destroyed"
    }


# CLI interface
if __name__ == "__main__":
    import asyncio
    import sys
    
    async def main():
        """CLI interface for SQLModel auto-migration."""
        if len(sys.argv) < 2:
            print("Usage: python -m app.database.sqlmodel_migration [init|info|sync|reset]")
            sys.exit(1)
        
        command = sys.argv[1]
        
        try:
            if command == "init":
                print("Initializing SQLModel tables...")
                result = await initialize_sqlmodel_tables()
                print(f"✅ Status: {result['status']}")
                if 'database_info' in result:
                    info = result['database_info']
                    print(f"Tables: {info['database_tables']} (DB) / {info['sqlmodel_tables']} (Models)")
            
            elif command == "info":
                print("Getting database information...")
                auto_migration = await get_auto_migration()
                info = await auto_migration.get_database_info()
                print(f"Environment: {info['environment']}")
                print(f"Auto-migration: {'Enabled' if info['auto_migration_enabled'] else 'Disabled'}")
                print(f"Database tables: {info['database_tables']}")
                print(f"SQLModel tables: {info['sqlmodel_tables']}")
                print(f"Extensions: {', '.join(info['extensions'])}")
            
            elif command == "sync":
                print("Syncing schema changes...")
                auto_migration = await get_auto_migration()
                result = await auto_migration.sync_schema_changes()
                print(f"✅ Status: {result['status']}")
                if result.get('changes'):
                    print("Changes:")
                    for change in result['changes']:
                        print(f"  - {change}")
            
            elif command == "reset":
                print("⚠️  Resetting development database...")
                result = await reset_development_database()
                print(f"✅ {result['status']}: {result.get('warning', 'Complete')}")
            
            else:
                print(f"Unknown command: {command}")
                sys.exit(1)
        
        except Exception as e:
            print(f"❌ Error: {str(e)}")
            sys.exit(1)
    
    asyncio.run(main())