"""Add retry system tables for error recovery and dead letter queue

Revision ID: 72d402eb3c32
Revises: 50121c073ff9
Create Date: 2025-10-04 19:10:16.720926

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = '72d402eb3c32'
down_revision: Union[str, Sequence[str], None] = '50121c073ff9'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    pass


def downgrade() -> None:
    """Downgrade schema."""
    pass
