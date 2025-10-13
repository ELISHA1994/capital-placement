"""Add search suggestions tables and materialized view

Revision ID: e01c68db2465
Revises: 1a047a9c676a
Create Date: 2025-10-13 12:23:23.940043

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'e01c68db2465'
down_revision: Union[str, Sequence[str], None] = '1a047a9c676a'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # Enable trigram extension for fuzzy prefix matching
    op.execute('CREATE EXTENSION IF NOT EXISTS pg_trgm;')

    # Create user_search_history table for personalized suggestions
    op.create_table(
        'user_search_history',
        sa.Column('id', sa.UUID(), nullable=False, server_default=sa.text('gen_random_uuid()')),
        sa.Column('tenant_id', sa.UUID(), nullable=False),
        sa.Column('user_id', sa.UUID(), nullable=False),
        sa.Column('query', sa.Text(), nullable=False),
        sa.Column('timestamp', sa.DateTime(timezone=True), nullable=False, server_default=sa.text('NOW()')),
        sa.Column('results_count', sa.Integer(), nullable=False, server_default=sa.text('0')),
        sa.Column('clicked', sa.Boolean(), nullable=False, server_default=sa.text('FALSE')),
        sa.PrimaryKeyConstraint('id'),
        sa.ForeignKeyConstraint(['tenant_id'], ['tenants.id'], ondelete='CASCADE'),
    )

    # Indexes for user search history
    # GIN index for fast prefix matching using trigrams
    op.execute("""
        CREATE INDEX idx_user_history_prefix
        ON user_search_history USING gin (query gin_trgm_ops)
    """)

    # Composite index for user-specific queries
    op.create_index(
        'idx_user_history_tenant_user_time',
        'user_search_history',
        ['tenant_id', 'user_id', sa.text('timestamp DESC')]
    )

    # Index for cleanup queries
    op.create_index(
        'idx_user_history_cleanup',
        'user_search_history',
        ['timestamp']
    )

    # Create materialized view for popular tenant searches
    op.execute("""
        CREATE MATERIALIZED VIEW tenant_popular_searches AS
        SELECT
            tenant_id,
            metadata->>'query' as query,
            COUNT(*) as frequency,
            MAX(timestamp) as last_used,
            AVG(CASE WHEN value > 0 THEN 1.0 ELSE 0.0 END) as success_rate
        FROM search_metrics
        WHERE metric_name = 'results_returned'
            AND timestamp > NOW() - INTERVAL '30 days'
            AND metadata->>'query' IS NOT NULL
            AND metadata->>'query' != ''
        GROUP BY tenant_id, metadata->>'query'
        HAVING COUNT(*) >= 2
        ORDER BY tenant_id, frequency DESC
    """)

    # Create unique index on materialized view
    op.create_index(
        'idx_popular_searches_tenant_query',
        'tenant_popular_searches',
        ['tenant_id', 'query'],
        unique=True
    )

    # Create performance index for frequency lookups
    op.execute("""
        CREATE INDEX idx_popular_searches_tenant_freq
        ON tenant_popular_searches(tenant_id, frequency DESC)
    """)


def downgrade() -> None:
    """Downgrade schema."""
    # Drop materialized view and its indexes
    op.execute('DROP MATERIALIZED VIEW IF EXISTS tenant_popular_searches CASCADE;')

    # Drop user_search_history table and its indexes
    op.drop_index('idx_user_history_cleanup', table_name='user_search_history')
    op.drop_index('idx_user_history_tenant_user_time', table_name='user_search_history')
    op.execute('DROP INDEX IF EXISTS idx_user_history_prefix')
    op.drop_table('user_search_history')

    # Note: We don't drop pg_trgm extension as it might be used by other tables
