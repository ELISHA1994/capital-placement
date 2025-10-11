"""create query expansions table

Revision ID: e3ee7c7e51f2
Revises: e84203120131
Create Date: 2025-10-10 23:20:00.000000

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "e3ee7c7e51f2"
down_revision: Union[str, Sequence[str], None] = 'e84203120131'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        "query_expansions",
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("tenant_id", sa.UUID(), nullable=False),
        sa.Column("original_query", sa.String(length=1000), nullable=False),
        sa.Column("query_hash", sa.String(length=128), nullable=False),
        sa.Column("expanded_terms", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("primary_skills", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("job_roles", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("experience_level", sa.String(length=100), nullable=True),
        sa.Column("industry", sa.String(length=100), nullable=True),
        sa.Column("confidence_score", sa.Float(), nullable=True),
        sa.Column("ai_model_used", sa.String(length=100), nullable=True),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("usage_count", sa.Integer(), nullable=False, server_default=sa.text("0")),
        sa.Column("last_used_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["tenant_id"], ["tenants.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("tenant_id", "query_hash", name="uq_query_expansions_tenant_hash"),
    )
    with op.batch_alter_table("query_expansions", schema=None) as batch_op:
        batch_op.create_index(batch_op.f("ix_query_expansions_tenant_id"), ["tenant_id"], unique=False)
        batch_op.create_index(batch_op.f("ix_query_expansions_original_query"), ["original_query"], unique=False)
        batch_op.create_index(batch_op.f("ix_query_expansions_expires_at"), ["expires_at"], unique=False)
        batch_op.create_index("ix_query_expansions_query_hash", ["query_hash"], unique=False)
        batch_op.create_index("ix_query_expansions_tenant_expires_at", ["tenant_id", "expires_at"], unique=False)


def downgrade() -> None:
    """Downgrade schema."""
    with op.batch_alter_table("query_expansions", schema=None) as batch_op:
        batch_op.drop_index("ix_query_expansions_tenant_expires_at")
        batch_op.drop_index("ix_query_expansions_query_hash")
        batch_op.drop_index(batch_op.f("ix_query_expansions_expires_at"))
        batch_op.drop_index(batch_op.f("ix_query_expansions_original_query"))
        batch_op.drop_index(batch_op.f("ix_query_expansions_tenant_id"))

    op.drop_table("query_expansions")

