import asyncio
from logging.config import fileConfig

from sqlalchemy import pool, engine_from_config
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import async_engine_from_config
from sqlmodel import SQLModel

from alembic import context

# CRITICAL: Import ALL models for autogenerate to detect schema changes
from app.infrastructure.persistence.models.auth_tables import UserTable, UserSessionTable, APIKeyTable
from app.infrastructure.persistence.models.profile_table import ProfileTable
from app.infrastructure.persistence.models.embedding_table import EmbeddingTable
from app.infrastructure.persistence.models.tenant_table import TenantTable
from app.infrastructure.persistence.models.audit_table import AuditLogTable
from app.infrastructure.persistence.models.retry_table import RetryStateModel, RetryAttemptModel, DeadLetterModel
from app.infrastructure.persistence.models.webhook_table import WebhookEndpointTable, WebhookDeliveryTable
from app.infrastructure.persistence.models.query_expansion_table import QueryExpansionTable

# this is the Alembic Config object, which provides
# access to the values within the .ini file in use.
config = context.config

# Interpret the config file for Python logging.
# This line sets up loggers basically.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# REQUIRED: SQLModel metadata for autogenerate support
target_metadata = SQLModel.metadata

# other values from the config, defined by the needs of env.py,
# can be acquired:
# my_important_option = config.get_main_option("my_important_option")
# ... etc.


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode.

    This configures the context with just a URL
    and not an Engine, though an Engine is acceptable
    here as well.  By skipping the Engine creation
    we don't even need a DBAPI to be available.

    Calls to context.execute() here emit the given string to the
    script output.

    """
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        compare_type=True,
        compare_server_default=True,
        render_as_batch=True
    )

    with context.begin_transaction():
        context.run_migrations()


def run_sync_migrations() -> None:
    """Run migrations in synchronous 'online' mode.
    
    This is used when calling Alembic from within an async context
    where we can't use asyncio.run().
    """
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        do_run_migrations(connection)

    connectable.dispose()


async def run_async_migrations() -> None:
    """Run migrations in async 'online' mode.
    
    This is used when running Alembic directly from command line.
    """
    connectable = async_engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)

    await connectable.dispose()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode.
    
    Detect the URL format and choose the appropriate method.
    """
    url = config.get_main_option("sqlalchemy.url")
    
    # Check if URL is for async operations (contains +asyncpg)
    if url and "asyncpg" in url:
        try:
            # Try to get the current event loop
            loop = asyncio.get_running_loop()
            # If we get here, we're in an async context, use sync version
            run_sync_migrations()
        except RuntimeError:
            # No event loop running, safe to use asyncio.run()
            asyncio.run(run_async_migrations())
    else:
        # Regular PostgreSQL URL, use sync version
        run_sync_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
