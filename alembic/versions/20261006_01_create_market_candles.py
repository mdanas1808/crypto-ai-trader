"""Create normalized historical market candles."""

from typing import Sequence, Union

import sqlalchemy as sa

from alembic import op

revision: str = "20261006_01"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "market_candles",
        sa.Column("exchange", sa.String(length=32), nullable=False),
        sa.Column("symbol", sa.String(length=32), nullable=False),
        sa.Column("interval", sa.String(length=8), nullable=False),
        sa.Column("open_time", sa.DateTime(timezone=True), nullable=False),
        sa.Column("open", sa.Numeric(precision=38, scale=18), nullable=False),
        sa.Column("high", sa.Numeric(precision=38, scale=18), nullable=False),
        sa.Column("low", sa.Numeric(precision=38, scale=18), nullable=False),
        sa.Column("close", sa.Numeric(precision=38, scale=18), nullable=False),
        sa.Column("volume", sa.Numeric(precision=38, scale=18), nullable=False),
        sa.PrimaryKeyConstraint(
            "exchange",
            "symbol",
            "interval",
            "open_time",
            name="pk_market_candles",
        ),
    )


def downgrade() -> None:
    op.drop_table("market_candles")
