import os
from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest

from app.analysis.eval_logger import FundamentalEvalLogger


@pytest.mark.asyncio
async def test_eval_logger_records_entry_and_logs(tmp_path):
    db_path = str(tmp_path / "test_eval.db")
    log_file = str(tmp_path / "test_eval.log")

    logger = FundamentalEvalLogger(db_path=db_path, log_file=log_file)

    eval_time = datetime(2024, 6, 1, 12, 0, tzinfo=timezone.utc)
    raw_xml = "<news_data>[1h ago | CoinDesk] Protocol bridge drained</news_data>"

    eval_id = logger.log_entry_evaluation(
        symbol="TAOUSDT",
        current_price=Decimal("450.50"),
        raw_inputs=raw_xml,
        verdict="REJECT",
        risk_level="CRITICAL",
        confidence=0.95,
        reason="Bridge drained of 50M funds.",
        metadata={"driving_news_id": "bridge_exploit_1", "verdict": "REJECT"},
        timestamp=eval_time,
    )

    assert eval_id == 1
    assert os.path.exists(log_file)
    with open(log_file, "r") as f:
        content = f.read()
    assert "ENTRY_EVAL" in content
    assert "TAOUSDT" in content
    assert "REJECT" in content

    record = logger.get_evaluation(eval_id)
    assert record is not None
    assert record["symbol"] == "TAOUSDT"
    assert record["current_price"] == 450.50
    assert record["verdict"] == "REJECT"
    assert record["risk_level"] == "CRITICAL"
    assert record["confidence"] == 0.95
    assert record["outcome_status"] == "PENDING"


@pytest.mark.asyncio
async def test_eval_logger_records_hypothetical_outcome_dodged_bullet(tmp_path):
    db_path = str(tmp_path / "test_eval.db")
    log_file = str(tmp_path / "test_eval.log")

    logger = FundamentalEvalLogger(db_path=db_path, log_file=log_file)

    eval_time = datetime(2024, 6, 1, 12, 0, tzinfo=timezone.utc)
    eval_id = logger.log_entry_evaluation(
        symbol="FETUSDT",
        current_price=Decimal("2.00"),
        raw_inputs="<news_data>Exploit</news_data>",
        verdict="REJECT",
        risk_level="CRITICAL",
        confidence=0.90,
        reason="Smart contract exploit.",
        timestamp=eval_time,
    )

    # Mock price fetcher returning declining prices (dodged bullet scenario)
    # T+1h: 1.80 (-10%), T+4h: 1.50 (-25%), T+24h: 1.00 (-50%)
    price_map = {
        eval_time + timedelta(hours=1): Decimal("1.80"),
        eval_time + timedelta(hours=4): Decimal("1.50"),
        eval_time + timedelta(hours=24): Decimal("1.00"),
    }

    async def mock_fetch_price(sym: str, target_time: datetime):
        return price_map.get(target_time)

    outcome = await logger.record_hypothetical_outcome(
        eval_id=eval_id,
        symbol="FETUSDT",
        entry_price=Decimal("2.00"),
        eval_time=eval_time,
        price_fetcher=mock_fetch_price,
        wait_if_live=False,
    )

    assert outcome["return_1h"] == pytest.approx(-0.10)
    assert outcome["return_4h"] == pytest.approx(-0.25)
    assert outcome["return_24h"] == pytest.approx(-0.50)
    assert "dodged a bullet" in outcome["assessment"]

    # Verify updated in SQLite
    record = logger.get_evaluation(eval_id)
    assert record["outcome_status"] == "COMPLETED"
    assert record["return_24h"] == pytest.approx(-0.50)
    assert record["price_t_24h"] == 1.00

    with open(log_file, "r") as f:
        log_content = f.read()
    assert "HYPOTHETICAL_OUTCOME" in log_content
    assert "dodged a bullet" in log_content


@pytest.mark.asyncio
async def test_eval_logger_records_hypothetical_outcome_missed_profit(tmp_path):
    db_path = str(tmp_path / "test_eval.db")
    log_file = str(tmp_path / "test_eval.log")

    logger = FundamentalEvalLogger(db_path=db_path, log_file=log_file)

    eval_time = datetime(2024, 6, 1, 12, 0, tzinfo=timezone.utc)
    eval_id = logger.log_entry_evaluation(
        symbol="NEARUSDT",
        current_price=Decimal("5.00"),
        raw_inputs="<news_data>FUD rumor</news_data>",
        verdict="REJECT",
        risk_level="HIGH",
        confidence=0.80,
        reason="FUD rumor vetoed entry.",
        timestamp=eval_time,
    )

    # Prices rallied after veto (missed profit scenario)
    # T+1h: 5.20 (+4%), T+4h: 5.50 (+10%), T+24h: 6.00 (+20%)
    price_map = {
        eval_time + timedelta(hours=1): Decimal("5.20"),
        eval_time + timedelta(hours=4): Decimal("5.50"),
        eval_time + timedelta(hours=24): Decimal("6.00"),
    }

    async def mock_fetch_price(sym: str, target_time: datetime):
        return price_map.get(target_time)

    outcome = await logger.record_hypothetical_outcome(
        eval_id=eval_id,
        symbol="NEARUSDT",
        entry_price=Decimal("5.00"),
        eval_time=eval_time,
        price_fetcher=mock_fetch_price,
        wait_if_live=False,
    )

    assert outcome["return_24h"] == pytest.approx(0.20)
    assert "missed a profitable trade" in outcome["assessment"]

    record = logger.get_evaluation(eval_id)
    assert record["outcome_status"] == "COMPLETED"
    assert record["return_24h"] == pytest.approx(0.20)


@pytest.mark.asyncio
async def test_eval_logger_logs_all_decisions(tmp_path):
    db_path = str(tmp_path / "test_eval_all.db")
    log_file = str(tmp_path / "test_eval_all.log")

    logger = FundamentalEvalLogger(db_path=db_path, log_file=log_file)
    eval_time = datetime(2024, 6, 1, 12, 0, tzinfo=timezone.utc)

    # 1. Log APPROVE (with news)
    logger.log_decision(
        symbol="SOLUSDT",
        current_price=150.0,
        raw_inputs="<news_data>[1h ago | CoinDesk] Positive developer update</news_data>",
        verdict="APPROVE",
        risk_level="LOW",
        confidence=0.9,
        reason="No adverse risks detected.",
        mode="ENTRY",
        timestamp=eval_time,
    )

    # 2. Log APPROVE (no-news)
    logger.log_decision(
        symbol="BTCUSDT",
        current_price=65000.0,
        raw_inputs="<news_data></news_data>",
        verdict="APPROVE",
        risk_level="LOW",
        confidence=0.5,
        reason="No recent headlines; approving entry.",
        mode="ENTRY",
        timestamp=eval_time,
    )

    # 3. Log POSITION_REVIEW HOLD
    logger.log_decision(
        symbol="FETUSDT",
        current_price=2.1,
        raw_inputs="<news_data>Routine commentary</news_data>",
        verdict="HOLD",
        risk_level="LOW",
        confidence=0.85,
        reason="No action required.",
        mode="POSITION_REVIEW",
        timestamp=eval_time,
    )

    # 4. Log POSITION_REVIEW REDUCE
    logger.log_decision(
        symbol="NEARUSDT",
        current_price=5.2,
        raw_inputs="<news_data>Upcoming unlock risk</news_data>",
        verdict="REDUCE",
        risk_level="HIGH",
        confidence=0.75,
        reason="Large token unlock approaching.",
        mode="POSITION_REVIEW",
        timestamp=eval_time,
    )

    # 5. Log POSITION_REVIEW EXIT
    logger.log_decision(
        symbol="TAOUSDT",
        current_price=400.0,
        raw_inputs="<news_data>Critical exploit</news_data>",
        verdict="EXIT",
        risk_level="CRITICAL",
        confidence=0.98,
        reason="Exploit drained funds.",
        mode="POSITION_REVIEW",
        timestamp=eval_time,
    )

    pending = logger.get_pending_evaluations()
    assert len(pending) == 5
    verdicts = [p["verdict"] for p in pending]
    assert "APPROVE" in verdicts
    assert "HOLD" in verdicts
    assert "REDUCE" in verdicts
    assert "EXIT" in verdicts


@pytest.mark.asyncio
async def test_compute_outcomes_script_processes_pending_evaluations(tmp_path):
    from scripts.compute_outcomes import compute_outcomes_for_database

    db_path = str(tmp_path / "test_eval_script.db")
    log_file = str(tmp_path / "test_eval_script.log")

    logger = FundamentalEvalLogger(db_path=db_path, log_file=log_file)
    eval_time = datetime(2024, 6, 1, 12, 0, tzinfo=timezone.utc)

    # Decision-time price was 100.0
    eval_id = logger.log_decision(
        symbol="SOLUSDT",
        current_price=100.0,
        raw_inputs="<news_data>News</news_data>",
        verdict="APPROVE",
        risk_level="LOW",
        confidence=0.9,
        reason="Approved signal",
        mode="ENTRY",
        timestamp=eval_time,
    )

    # NEXT candle at 12:00 has OPEN price = 102.0 (hypothetical fill price!)
    # T+1h candle (13:00) has close = 107.1 (+5.0% from 102.0)
    # T+4h candle (16:00) has close = 112.2 (+10.0% from 102.0)
    # T+24h candle (next day 12:00) has close = 122.4 (+20.0% from 102.0)
    mock_candles = [
        # open_time_ms, open, high, low, close, volume, close_time_ms
        [
            int(eval_time.timestamp() * 1000),
            "102.0",
            "103.0",
            "101.0",
            "102.5",
            "100",
            int((eval_time + timedelta(hours=1)).timestamp() * 1000) - 1,
        ],
        [
            int((eval_time + timedelta(hours=1)).timestamp() * 1000),
            "102.5",
            "108.0",
            "102.0",
            "107.1",
            "100",
            int((eval_time + timedelta(hours=2)).timestamp() * 1000) - 1,
        ],
        [
            int((eval_time + timedelta(hours=4)).timestamp() * 1000),
            "110.0",
            "113.0",
            "109.0",
            "112.2",
            "100",
            int((eval_time + timedelta(hours=5)).timestamp() * 1000) - 1,
        ],
        [
            int((eval_time + timedelta(hours=24)).timestamp() * 1000),
            "120.0",
            "125.0",
            "119.0",
            "122.4",
            "100",
            int((eval_time + timedelta(hours=25)).timestamp() * 1000) - 1,
        ],
    ]

    async def mock_candle_fetcher(sym: str, start_time: datetime, limit: int):
        return mock_candles

    processed = await compute_outcomes_for_database(
        db_path=db_path,
        candle_fetcher=mock_candle_fetcher,
    )

    assert processed == 1

    record = logger.get_evaluation(eval_id)
    assert record["outcome_status"] == "COMPLETED"
    # Hypothetical fill price is next candle open (102.0), not decision price (100.0)
    assert record["hypothetical_fill_price"] == pytest.approx(102.0)
    assert record["price_t_1h"] == pytest.approx(107.1)
    assert record["return_1h"] == pytest.approx(0.05)
    assert record["price_t_4h"] == pytest.approx(112.2)
    assert record["return_4h"] == pytest.approx(0.10)
    assert record["price_t_24h"] == pytest.approx(122.4)
    assert record["return_24h"] == pytest.approx(0.20)


@pytest.mark.asyncio
async def test_eval_logger_records_blocked_by_regime_and_computes_outcomes(tmp_path):
    from scripts.compute_outcomes import compute_outcomes_for_database

    db_path = str(tmp_path / "test_regime_eval.db")
    log_file = str(tmp_path / "test_regime_eval.log")

    logger = FundamentalEvalLogger(db_path=db_path, log_file=log_file)
    eval_time = datetime(2024, 6, 1, 12, 0, tzinfo=timezone.utc)

    # Log BLOCKED_BY_REGIME signal
    eval_id = logger.log_regime_blocked_signal(
        symbol="TAOUSDT",
        current_price=Decimal("400.00"),
        reason="Blocked: BTC downtrend regime",
        timestamp=eval_time,
    )

    rec = logger.get_evaluation(eval_id)
    assert rec is not None
    assert rec["symbol"] == "TAOUSDT"
    assert rec["verdict"] == "APPROVE"
    assert rec["regime_state"] == "DOWNTREND"
    assert rec["regime_blocked"] == 1
    assert rec["outcome_status"] == "PENDING"

    # Mock candles: Next open=400.0, T+1h=392.0 (-2%), T+4h=380.0 (-5%), T+24h=360.0 (-10%)
    mock_candles = [
        [
            int(eval_time.timestamp() * 1000),
            "400.0",
            "405.0",
            "390.0",
            "395.0",
            "100",
            int((eval_time + timedelta(hours=1)).timestamp() * 1000) - 1,
        ],
        [
            int((eval_time + timedelta(hours=1)).timestamp() * 1000),
            "395.0",
            "396.0",
            "390.0",
            "392.0",
            "100",
            int((eval_time + timedelta(hours=2)).timestamp() * 1000) - 1,
        ],
        [
            int((eval_time + timedelta(hours=4)).timestamp() * 1000),
            "385.0",
            "388.0",
            "378.0",
            "380.0",
            "100",
            int((eval_time + timedelta(hours=5)).timestamp() * 1000) - 1,
        ],
        [
            int((eval_time + timedelta(hours=24)).timestamp() * 1000),
            "365.0",
            "370.0",
            "358.0",
            "360.0",
            "100",
            int((eval_time + timedelta(hours=25)).timestamp() * 1000) - 1,
        ],
    ]

    async def mock_candle_fetcher(sym: str, start_time: datetime, limit: int):
        return mock_candles

    processed = await compute_outcomes_for_database(
        db_path=db_path,
        candle_fetcher=mock_candle_fetcher,
    )
    assert processed == 1

    updated = logger.get_evaluation(eval_id)
    assert updated["outcome_status"] == "COMPLETED"
    assert updated["hypothetical_fill_price"] == pytest.approx(400.0)
    assert updated["return_1h"] == pytest.approx(-0.02)
    assert updated["return_4h"] == pytest.approx(-0.05)
    assert updated["return_24h"] == pytest.approx(-0.10)
    # Net returns for Satellite coins deduct 0.60% (0.006)
    assert updated["net_return_1h"] == pytest.approx(-0.026)
    assert updated["net_return_4h"] == pytest.approx(-0.056)
    assert updated["net_return_24h"] == pytest.approx(-0.106)


def test_eval_logger_fallback_field_and_queries(tmp_path):
    db_path = str(tmp_path / "test_fallback.db")
    log_file = str(tmp_path / "test_fallback.log")
    logger = FundamentalEvalLogger(db_path=db_path, log_file=log_file)

    eval_time = datetime(2024, 6, 1, 12, 0, tzinfo=timezone.utc)
    id1 = logger.log_decision(
        symbol="FETUSDT",
        current_price=2.0,
        raw_inputs="news",
        verdict="FALLBACK_REJECT",
        risk_level="HIGH",
        confidence=0.0,
        reason="Fallback rejection due to network error",
        mode="ENTRY",
        is_fallback=True,
        timestamp=eval_time,
    )
    id2 = logger.log_decision(
        symbol="FETUSDT",
        current_price=2.0,
        raw_inputs="news",
        verdict="REJECT",
        risk_level="HIGH",
        confidence=0.8,
        reason="Exploit detected",
        mode="ENTRY",
        is_fallback=False,
        timestamp=eval_time,
    )

    rec1 = logger.get_evaluation(id1)
    rec2 = logger.get_evaluation(id2)

    assert rec1["is_fallback"] == 1
    assert rec1["verdict"] == "FALLBACK_REJECT"
    assert rec2["is_fallback"] == 0
    assert rec2["verdict"] == "REJECT"
