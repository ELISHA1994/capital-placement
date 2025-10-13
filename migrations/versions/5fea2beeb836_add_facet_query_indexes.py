"""add_facet_query_indexes

Revision ID: 5fea2beeb836
Revises: 1c4c2a91968c
Create Date: 2025-10-13 11:17:09.251662

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '5fea2beeb836'
down_revision: Union[str, Sequence[str], None] = '1c4c2a91968c'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    pass


def downgrade() -> None:
    """Downgrade schema."""
    pass
