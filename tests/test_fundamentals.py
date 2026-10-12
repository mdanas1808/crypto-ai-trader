from datetime import datetime, timezone
from unittest.mock import AsyncMock

import httpx
import pytest

from app.analysis.fundamentals import (
    CRYPTOCOMPARE_NEWS_BASE_URL,
    NO_NEWS_FALLBACK,
    NewsFetchError,
    fetch_recent_news,
    fetch_recent_news_items,
    format_news_xml,
)


@pytest.mark.asyncio
async def test_fetch_recent_news_success_formats_xml_blocks():
    fixed_now = datetime(2024, 6, 1, 12, 0, tzinfo=timezone.utc)
    two_hours_ago = 1717236000  # 2024-06-01 10:00:00 UTC (2h ago)
    four_hours_ago = 1717228800  # 2024-06-01 08:00:00 UTC (4h ago)

    payload = {
        "Data": [
            {
                "published_on": two_hours_ago,
                "source_info": {"name": "CoinDesk"},
                "title": "Fetch.ai Announces ASI Alliance Upgrade",
                "body": "Decentralized AI network completes major token merger milestone.",
            },
            {
                "published_on": four_hours_ago,
                "source": "Cointelegraph",
                "title": "Institutional AI Demand Rises",
                "body": "Compute cluster utilization hits new quarterly record.",
            },
            {
                "published_on": two_hours_ago,
                "title": "Third Headline Ignored When Limit Is Two",
                "body": "Should not appear when limit=2.",
            },
        ]
    }
    mock_client = AsyncMock()
    mock_client.get = AsyncMock(return_value=httpx.Response(status_code=200, json=payload))

    result = await fetch_recent_news("FETUSDT", limit=2, async_client=mock_client, now=fixed_now)

    assert mock_client.get.await_count == 1
    called_url = mock_client.get.call_args.args[0]
    assert called_url == f"{CRYPTOCOMPARE_NEWS_BASE_URL}?categories=FET&lang=EN"
    assert "<news_data>" in result
    assert "</news_data>" in result
    assert (
        "[2h ago | CoinDesk] Fetch.ai Announces ASI Alliance Upgrade :: Decentralized AI network completes major token merger milestone."
        in result
    )
    assert (
        "[4h ago | Cointelegraph] Institutional AI Demand Rises :: Compute cluster utilization hits new quarterly record."
        in result
    )
    assert "Third Headline Ignored" not in result


@pytest.mark.asyncio
async def test_fetch_recent_news_items_truncates_summary_to_300_chars():
    fixed_now = datetime(2024, 6, 1, 12, 0, tzinfo=timezone.utc)
    long_body = "A" * 500

    payload = {
        "Data": [
            {
                "published_on": 1717236000,
                "source_info": {"name": "CryptoGlobe"},
                "title": "Protocol Update",
                "body": long_body,
            }
        ]
    }
    mock_client = AsyncMock()
    mock_client.get = AsyncMock(return_value=httpx.Response(status_code=200, json=payload))

    items = await fetch_recent_news_items(
        "TAOUSDT", limit=1, async_client=mock_client, now=fixed_now
    )
    assert len(items) == 1
    assert items[0]["age_hours"] == 2
    assert items[0]["source"] == "CryptoGlobe"
    assert items[0]["headline"] == "Protocol Update"
    assert len(items[0]["summary"]) == 300
    assert items[0]["summary"] == "A" * 300


@pytest.mark.asyncio
async def test_fetch_recent_news_raises_on_network_failure():
    mock_client = AsyncMock()
    mock_client.get = AsyncMock(side_effect=httpx.ConnectError("DNS lookup failed"))

    with pytest.raises(NewsFetchError):
        await fetch_recent_news("TAOUSDT", async_client=mock_client)


@pytest.mark.asyncio
async def test_fetch_recent_news_raises_on_timeout():
    mock_client = AsyncMock()
    mock_client.get = AsyncMock(side_effect=httpx.TimeoutException("Request timed out after 3.0s"))

    with pytest.raises(NewsFetchError):
        await fetch_recent_news("RENDERUSDT", async_client=mock_client)


@pytest.mark.asyncio
async def test_fetch_recent_news_raises_on_non_200_status():
    mock_client = AsyncMock()
    mock_client.get = AsyncMock(
        return_value=httpx.Response(status_code=503, text="Service Unavailable")
    )

    with pytest.raises(NewsFetchError):
        await fetch_recent_news("NEARUSDT", async_client=mock_client)


@pytest.mark.asyncio
async def test_fetch_recent_news_fallback_on_empty_payload():
    mock_client = AsyncMock()
    mock_client.get = AsyncMock(return_value=httpx.Response(status_code=200, json={"Data": []}))

    result = await fetch_recent_news("INJUSDT", async_client=mock_client)
    assert result == "<news_data></news_data>"


@pytest.mark.asyncio
async def test_fetch_recent_news_handles_empty_symbol_or_non_positive_limit():
    mock_client = AsyncMock()
    assert await fetch_recent_news("", async_client=mock_client) == NO_NEWS_FALLBACK
    assert await fetch_recent_news("BTCUSDT", limit=0, async_client=mock_client) == NO_NEWS_FALLBACK
    assert mock_client.get.await_count == 0


def test_format_news_xml_empty():
    assert format_news_xml([]) == "<news_data></news_data>"


@pytest.mark.asyncio
async def test_fetch_recent_news_filters_unmatched_categories_and_tags():
    fixed_now = datetime(2024, 6, 1, 12, 0, tzinfo=timezone.utc)
    ts = 1717236000

    payload = {
        "Data": [
            {
                "published_on": ts,
                "categories": "BTC|BLOCKCHAIN",
                "title": "Bitcoin Hits New Milestone",
                "body": "BTC network activity increases.",
            },
            {
                "published_on": ts,
                "categories": "ETH|DEFI",
                "title": "Ethereum Gas Fees Drop",
                "body": "Unrelated Ethereum article categorized under ETH.",
            },
            {
                "published_on": ts,
                "tags": "SOL|SOLANA",
                "title": "Solana Ecosystem Expansion",
                "body": "SOL developers announce hackathon.",
            },
        ]
    }
    mock_client = AsyncMock()
    mock_client.get = AsyncMock(return_value=httpx.Response(status_code=200, json=payload))

    # Fetch news for SOLUSDT: ETH article should be filtered out, SOL kept
    sol_items = await fetch_recent_news_items(
        "SOLUSDT", limit=10, async_client=mock_client, now=fixed_now
    )
    assert len(sol_items) == 1
    assert sol_items[0]["headline"] == "Solana Ecosystem Expansion"


@pytest.mark.asyncio
async def test_fetch_recent_news_discards_substring_headlines():
    fixed_now = datetime(2024, 6, 1, 12, 0, tzinfo=timezone.utc)
    ts = 1717236000

    payload = {
        "Data": [
            {
                "published_on": ts,
                "title": "Government Passes Resolution On Solar Energy",  # 'sol' inside 'resolution', 'solar'
                "body": "New council resolution passed today.",
            },
            {
                "published_on": ts,
                "title": "Solana DEX Volume Flips Ethereum",  # Standalone 'Solana'
                "body": "Decentralized exchange volume breaks records.",
            },
            {
                "published_on": ts,
                "title": "Bank Consolidation Accelerates",  # 'sol' inside 'consolidation'
                "body": "Regional banking sector consolidates.",
            },
            {
                "published_on": ts,
                "title": "SOL Breaks Key Resistance Level",  # Standalone 'SOL'
                "body": "Technical breakout confirmed above $150.",
            },
        ]
    }
    mock_client = AsyncMock()
    mock_client.get = AsyncMock(return_value=httpx.Response(status_code=200, json=payload))

    items = await fetch_recent_news_items(
        "SOLUSDT", limit=10, async_client=mock_client, now=fixed_now
    )
    headlines = [item["headline"] for item in items]
    assert "Solana DEX Volume Flips Ethereum" in headlines
    assert "SOL Breaks Key Resistance Level" in headlines
    assert "Government Passes Resolution On Solar Energy" not in headlines
    assert "Bank Consolidation Accelerates" not in headlines


@pytest.mark.asyncio
async def test_fetch_recent_news_matches_cashtags_and_pairs_without_canonical_tags():
    fixed_now = datetime(2024, 6, 1, 12, 0, tzinfo=timezone.utc)
    ts = 1717236000

    payload = {
        "Data": [
            {
                "published_on": ts,
                "title": "$SOL Surges Past $180 Resistance",
                "body": "Trading volume soars.",
            },
            {
                "published_on": ts,
                "title": "SOL/USDT Perpetual Volume Reaches Record High",
                "body": "Exchange volumes increase.",
            },
            {
                "published_on": ts,
                "categories": "BTC|CRYPTO",  # Missing canonical tag SOL, but regex matches
                "title": "SOL Breaks All-Time High",
                "body": "Solana hits new record.",
            },
        ]
    }
    mock_client = AsyncMock()
    mock_client.get = AsyncMock(return_value=httpx.Response(status_code=200, json=payload))

    items = await fetch_recent_news_items(
        "SOLUSDT", limit=10, async_client=mock_client, now=fixed_now
    )
    headlines = [item["headline"] for item in items]
    assert "$SOL Surges Past $180 Resistance" in headlines
    assert "SOL/USDT Perpetual Volume Reaches Record High" in headlines
    assert "SOL Breaks All-Time High" in headlines


@pytest.mark.asyncio
async def test_fetch_recent_news_logs_discarded_articles(caplog):
    fixed_now = datetime(2024, 6, 1, 12, 0, tzinfo=timezone.utc)
    ts = 1717236000

    payload = {
        "Data": [
            {
                "published_on": ts,
                "title": "Government Passes Resolution On Solar Energy",
                "body": "Solar energy subsidy passed.",
            },
            {
                "published_on": ts,
                "categories": "ETH|DEFI",
                "title": "Ethereum Gas Fees Drop",
                "body": "Unrelated Ethereum article.",
            },
        ]
    }
    mock_client = AsyncMock()
    mock_client.get = AsyncMock(return_value=httpx.Response(status_code=200, json=payload))

    import logging

    with caplog.at_level(logging.DEBUG, logger="app.analysis.fundamentals"):
        items = await fetch_recent_news_items(
            "SOLUSDT", limit=10, async_client=mock_client, now=fixed_now
        )
        assert len(items) == 0
        assert "Discarded article: Government Passes Resolution On Solar Energy" in caplog.text
        assert "Discarded article: Ethereum Gas Fees Drop" in caplog.text


@pytest.mark.asyncio
async def test_strict_ticker_matching_ambiguous_assets(caplog):
    fixed_now = datetime(2024, 6, 1, 12, 0, tzinfo=timezone.utc)
    ts = 1717236000

    payload = {
        "Data": [
            {
                "published_on": ts,
                "title": "Bitcoin is near all-time high as ETF inflows surge",  # Lowercase 'near'
                "body": "BTC approaches key resistance near 70k.",
            },
            {
                "published_on": ts,
                "title": "$NEAR Surges 15% on Ecosystem Grant Expansion",  # Cashtag $NEAR
                "body": "Grant program announced.",
            },
            {
                "published_on": ts,
                "title": "Near Protocol Announces New Sharding Phase",  # Canonical name
                "body": "Layer 1 blockchain upgrades throughput.",
            },
            {
                "published_on": ts,
                "title": "NEAR Breaks Resistance Above $7.50",  # Exact uppercase NEAR
                "body": "Strong breakout confirmed.",
            },
        ]
    }
    mock_client = AsyncMock()
    mock_client.get = AsyncMock(return_value=httpx.Response(status_code=200, json=payload))

    import logging

    with caplog.at_level(logging.DEBUG, logger="app.analysis.fundamentals"):
        items = await fetch_recent_news_items(
            "NEARUSDT", limit=10, async_client=mock_client, now=fixed_now
        )
        headlines = [item["headline"] for item in items]
        assert "Bitcoin is near all-time high as ETF inflows surge" not in headlines
        assert "$NEAR Surges 15% on Ecosystem Grant Expansion" in headlines
        assert "Near Protocol Announces New Sharding Phase" in headlines
        assert "NEAR Breaks Resistance Above $7.50" in headlines
        assert (
            "Discarded article: Bitcoin is near all-time high as ETF inflows surge" in caplog.text
        )
