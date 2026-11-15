"""Add leveraged fields to paper_trades and paper_signals tables.

Revision ID: 20261115_01
Revises: 20261006_01
Create Date: 2026-11-15 13:00:00.000000
"""

from typing import Sequence, Union

import sqlalchemy as sa

from alembic import op

revision: str = "20261115_01"
down_revision: Union[str, None] = "20261006_01"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # 1. Ensure paper_trades exists or alter columns
    bind = op.get_bind()
    insp = sa.inspect(bind)
    tables = insp.get_table_names()

    if "paper_trades" in tables:
        cols = [c["name"] for c in insp.get_columns("paper_trades")]
        if "direction" not in cols:
            op.add_column(
                "paper_trades",
                sa.Column("direction", sa.String(16), nullable=False, server_default="LONG"),
            )
        if "leverage" not in cols:
            op.add_column(
                "paper_trades",
                sa.Column("leverage", sa.Integer(), nullable=False, server_default="1"),
            )
        if "margin_used" not in cols:
            op.add_column(
                "paper_trades",
                sa.Column("margin_used", sa.Numeric(38, 18), nullable=False, server_default="0"),
            )
        if "stop_loss" not in cols:
            op.add_column(
                "paper_trades",
                sa.Column("stop_loss", sa.Numeric(38, 18), nullable=False, server_default="0"),
            )
        if "take_profit" not in cols:
            op.add_column(
                "paper_trades",
                sa.Column("take_profit", sa.Numeric(38, 18), nullable=False, server_default="0"),
            )
        if "realized_pnl" not in cols:
            op.add_column(
                "paper_trades",
                sa.Column("realized_pnl", sa.Numeric(38, 18), nullable=False, server_default="0"),
            )
    else:
        op.create_table(
            "paper_trades",
            sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
            sa.Column("exchange", sa.String(32), nullable=False, server_default="binance_spot"),
            sa.Column("symbol", sa.String(32), nullable=False),
            sa.Column("side", sa.String(16), nullable=False, server_default="long"),
            sa.Column("direction", sa.String(16), nullable=False, server_default="LONG"),
            sa.Column("leverage", sa.Integer(), nullable=False, server_default="1"),
            sa.Column("margin_used", sa.Numeric(38, 18), nullable=False, server_default="0"),
            sa.Column("stop_loss", sa.Numeric(38, 18), nullable=False, server_default="0"),
            sa.Column("take_profit", sa.Numeric(38, 18), nullable=False, server_default="0"),
            sa.Column("realized_pnl", sa.Numeric(38, 18), nullable=False, server_default="0"),
            sa.Column("entry_time", sa.DateTime(timezone=True), nullable=False),
            sa.Column("exit_time", sa.DateTime(timezone=True), nullable=False),
            sa.Column("entry_price", sa.Numeric(38, 18), nullable=False),
            sa.Column("exit_price", sa.Numeric(38, 18), nullable=False),
            sa.Column("quantity", sa.Numeric(38, 18), nullable=False),
            sa.Column("pnl", sa.Numeric(38, 18), nullable=False),
            sa.Column("exit_reason", sa.String(64), nullable=False),
        )

    # 2. Ensure paper_signals exists or alter columns
    if "paper_signals" in tables:
        cols = [c["name"] for c in insp.get_columns("paper_signals")]
        if "direction" not in cols:
            op.add_column(
                "paper_signals",
                sa.Column("direction", sa.String(16), nullable=False, server_default="LONG"),
            )
        if "leverage" not in cols:
            op.add_column(
                "paper_signals",
                sa.Column("leverage", sa.Integer(), nullable=False, server_default="1"),
            )
        if "margin_used" not in cols:
            op.add_column(
                "paper_signals",
                sa.Column("margin_used", sa.Numeric(38, 18), nullable=False, server_default="0"),
            )
        if "stop_loss" not in cols:
            op.add_column(
                "paper_signals",
                sa.Column("stop_loss", sa.Numeric(38, 18), nullable=False, server_default="0"),
            )
        if "take_profit" not in cols:
            op.add_column(
                "paper_signals",
                sa.Column("take_profit", sa.Numeric(38, 18), nullable=False, server_default="0"),
            )
    else:
        op.create_table(
            "paper_signals",
            sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
            sa.Column("exchange", sa.String(32), nullable=False, server_default="binance_spot"),
            sa.Column("symbol", sa.String(32), nullable=False),
            sa.Column("side", sa.String(16), nullable=False, server_default="long"),
            sa.Column("direction", sa.String(16), nullable=False, server_default="LONG"),
            sa.Column("leverage", sa.Integer(), nullable=False, server_default="1"),
            sa.Column("margin_used", sa.Numeric(38, 18), nullable=False, server_default="0"),
            sa.Column("stop_loss", sa.Numeric(38, 18), nullable=False, server_default="0"),
            sa.Column("take_profit", sa.Numeric(38, 18), nullable=False, server_default="0"),
            sa.Column("decision_time", sa.DateTime(timezone=True), nullable=False),
            sa.Column("signal_open_time", sa.DateTime(timezone=True), nullable=False),
            sa.Column("atr_14", sa.Numeric(38, 18), nullable=False),
            sa.Column("explanation", sa.Text(), nullable=False),
        )


def downgrade() -> None:
    for col in ("direction", "leverage", "margin_used", "stop_loss", "take_profit", "realized_pnl"):
        try:
            op.drop_column("paper_trades", col)
        except Exception:
            pass
    for col in ("direction", "leverage", "margin_used", "stop_loss", "take_profit"):
        try:
            op.drop_column("paper_signals", col)
        except Exception:
            pass
