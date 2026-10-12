"""Fundamental news ingestion for AI risk evaluation.

Queries public CryptoCompare news endpoints for asset-specific headlines
and short summaries with strict timeouts and deterministic fallback.
Captures age in hours, publisher name, headline, and summary truncated to 300 chars,
and formats them into structured <news_data> XML blocks.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from typing import Any

import httpx

logger = logging.getLogger(__name__)

CRYPTOCOMPARE_NEWS_BASE_URL = "https://min-api.cryptocompare.com/data/v2/news/"
DEFAULT_NEWS_TIMEOUT = 3.0
NO_NEWS_FALLBACK = "<news_data></news_data>"


class NewsFetchError(Exception):
    """Raised when news fetching encounters network or HTTP failures."""


ASSET_NAMES: dict[str, list[str]] = {
    "BTC": ["Bitcoin"],
    "ETH": ["Ethereum", "Ether"],
    "SOL": ["Solana"],
    "BNB": ["Binance"],
    "ZEC": ["Zcash"],
    "TAO": ["Bittensor"],
    "FET": ["Fetch", "Fetch.ai", "ASI"],
    "RENDER": ["Render", "Render Network"],
    "NEAR": ["NEAR Protocol", "Near Protocol"],
    "INJ": ["Injective", "Injective Protocol"],
    "LINK": ["Chainlink"],
    "OP": ["Optimism"],
    "ONE": ["Harmony", "Harmony ONE"],
    "GAS": ["Neo Gas", "Gas token", "GAS token"],
}


def _matches_symbol_regex(text: str, ticker: str, symbol: str) -> bool:
    """Universal strict ticker matching for ALL symbols.

    Requires:
    1. Exact case-sensitive uppercase ticker: e.g. 'BTC', 'SOL', 'NEAR', 'FET'
    2. Cashtag: e.g. '$BTC', '$sol', '$NEAR'
    3. Pair notation: e.g. 'BTCUSDT', 'SOL/USDT', 'NEAR-USD'
    4. Canonical project name(s): e.g. 'Bitcoin', 'Solana', 'Fetch.ai', 'Render'
    Discards any lowercase or ordinary prose matches for all assets.
    """
    if not text or not ticker:
        return False

    ticker_upper = ticker.upper()

    # 1. Exact uppercase ticker (case-sensitive!): e.g. "SOL", "SOLUSDT", "SOL/USDT"
    exact_ticker_pattern = rf"(?:^|\W){re.escape(ticker_upper)}(?:USDT|USD|-USD|/USDT|/USD|\W|$)"
    if re.search(exact_ticker_pattern, text):
        return True

    # 2. Cashtag (e.g. $SOL, $sol, $NEAR):
    cashtag_pattern = rf"\${re.escape(ticker_upper)}(?:USDT|USD|-USD|\W|$)"
    if re.search(cashtag_pattern, text, re.IGNORECASE):
        return True

    # 3. Canonical project names (e.g. "Solana", "Bitcoin", "Ethereum", "Fetch.ai", "Render"):
    canonical_names = ASSET_NAMES.get(ticker_upper, [])
    if canonical_names:
        name_patterns = "|".join(re.escape(n) for n in canonical_names)
        canonical_pattern = rf"(?:^|\W)(?:{name_patterns})(?:\W|$)"
        if re.search(canonical_pattern, text, re.IGNORECASE):
            return True

    # 4. Explicit trading pair with symbol or quote (e.g. "SOL/USDT", "sol-usd", "SOLUSDT"):
    pair_pattern = rf"(?:^|\W){re.escape(ticker_upper)}(?:USDT|USD|-USD|/USDT|/USD)(?:\W|$)"
    if re.search(pair_pattern, text, re.IGNORECASE):
        return True

    return False


def _ticker_only_substring_of_word(text: str, ticker: str, symbol: str) -> bool:
    """Check if the ticker appears in text ONLY as a substring of other words,
    without any standalone match for the ticker, symbol, cashtag, or asset names.
    """
    if not text or not ticker:
        return False
    ticker_lower = ticker.lower()
    text_lower = text.lower()
    if ticker_lower not in text_lower:
        return False
    if _matches_symbol_regex(text, ticker, symbol):
        return False
    return True


def _matches_category_or_tags(item: dict[str, Any], ticker: str, symbol: str) -> bool | None:
    """Check if CryptoCompare categories or tag lists include the relevant asset/ticker.
    Returns True if matched, False if categories/tags exist but did not match,
    or None if no categories/tags were present.
    """
    ticker_upper = ticker.upper()
    symbol_upper = symbol.upper()
    valid_targets = {ticker_upper, symbol_upper}
    for name in ASSET_NAMES.get(ticker_upper, []):
        valid_targets.add(name.upper())

    cat_tokens: set[str] = set()

    cats = item.get("categories")
    if isinstance(cats, str) and cats.strip():
        cat_tokens.update(part.strip().upper() for part in re.split(r"[|,]", cats) if part.strip())
    elif isinstance(cats, (list, tuple, set)):
        cat_tokens.update(str(part).strip().upper() for part in cats if str(part).strip())

    tags = item.get("tags")
    if isinstance(tags, str) and tags.strip():
        cat_tokens.update(part.strip().upper() for part in re.split(r"[|,]", tags) if part.strip())
    elif isinstance(tags, (list, tuple, set)):
        cat_tokens.update(str(part).strip().upper() for part in tags if str(part).strip())

    if cat_tokens:
        return any(target in cat_tokens for target in valid_targets)

    return None


def _extract_ticker(symbol: str) -> str:
    """Strip 'USDT' quote suffix from a trading symbol to get the base ticker."""
    if not isinstance(symbol, str):
        return ""
    cleaned = symbol.strip().upper()
    if cleaned.endswith("USDT") and len(cleaned) > 4:
        return cleaned[:-4]
    return cleaned


def _calculate_age_hours(published_val: Any, now: datetime) -> int:
    """Calculate age in hours relative to current UTC time."""
    if published_val is None:
        return 0
    try:
        if isinstance(published_val, (int, float)):
            pub_dt = datetime.fromtimestamp(float(published_val), tz=timezone.utc)
        elif isinstance(published_val, str):
            clean = published_val.strip()
            if clean.isdigit() or (
                clean.replace(".", "", 1).isdigit() and not clean.startswith("-")
            ):
                pub_dt = datetime.fromtimestamp(float(clean), tz=timezone.utc)
            else:
                pub_dt = datetime.fromisoformat(clean.replace("Z", "+00:00"))
        elif isinstance(published_val, datetime):
            pub_dt = (
                published_val
                if published_val.tzinfo is not None
                else published_val.replace(tzinfo=timezone.utc)
            )
        else:
            return 0
        diff_seconds = (now - pub_dt).total_seconds()
        return max(0, int(diff_seconds // 3600))
    except Exception:
        return 0


def _extract_source_name(item: dict[str, Any]) -> str:
    """Extract publisher/source name from article dict."""
    source_info = item.get("source_info")
    if isinstance(source_info, dict):
        name = str(source_info.get("name") or source_info.get("source") or "").strip()
        if name:
            return name
    name = str(item.get("source") or item.get("publisher") or "").strip()
    return name if name else "Unknown"


def format_news_xml(items: list[dict[str, Any]]) -> str:
    """Format news items into a structured <news_data> XML block matching:
    [{age_hours}h ago | {source}] {headline} :: {summary_truncated_to_300_chars}
    """
    if not items:
        return "<news_data></news_data>"

    lines: list[str] = []
    for item in items:
        age_hours = item.get("age_hours", 0)
        source = item.get("source", "Unknown")
        headline = item.get("headline", "")
        summary = item.get("summary", "")
        if headline and summary:
            lines.append(f"[{age_hours}h ago | {source}] {headline} :: {summary}")
        elif headline:
            lines.append(f"[{age_hours}h ago | {source}] {headline}")
        elif summary:
            lines.append(f"[{age_hours}h ago | {source}] {summary}")

    if not lines:
        return "<news_data></news_data>"

    return "<news_data>\n" + "\n".join(lines) + "\n</news_data>"


async def fetch_recent_news_items(
    symbol: str,
    limit: int = 10,
    *,
    timeout: float = DEFAULT_NEWS_TIMEOUT,
    async_client: httpx.AsyncClient | None = None,
    now: datetime | None = None,
) -> list[dict[str, Any]]:
    """Fetch structured news items for a symbol from CryptoCompare."""
    ticker = _extract_ticker(symbol)
    if not ticker or limit <= 0:
        return []

    url = f"{CRYPTOCOMPARE_NEWS_BASE_URL}?categories={ticker}&lang=EN"
    ref_time = now if now is not None else datetime.now(timezone.utc)

    try:
        if async_client is not None:
            response = await async_client.get(url, timeout=timeout)
        else:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.get(url)

        if response.status_code != 200:
            logger.warning(
                "CryptoCompare news API returned HTTP %s for %s",
                response.status_code,
                ticker,
            )
            raise NewsFetchError(
                f"CryptoCompare news API returned HTTP {response.status_code} for {ticker}"
            )

        payload: Any = response.json()
        articles: list[Any] = []
        if isinstance(payload, dict):
            raw_data = (
                payload.get("Data")
                if "Data" in payload
                else (payload.get("data") or payload.get("articles") or [])
            )
            if isinstance(raw_data, list):
                articles = raw_data
        elif isinstance(payload, list):
            articles = payload

        if not articles:
            return []

        items: list[dict[str, Any]] = []
        for item in articles:
            if not isinstance(item, dict):
                continue

            pub_val = (
                item.get("published_on")
                or item.get("published_at")
                or item.get("time")
                or item.get("timestamp")
            )
            age_hours = _calculate_age_hours(pub_val, ref_time)
            source = _extract_source_name(item)
            headline = str(item.get("title") or item.get("headline") or "").strip()
            body = str(
                item.get("body") or item.get("summary") or item.get("description") or ""
            ).strip()
            summary = body[:300].strip() if len(body) > 300 else body

            if not headline and not summary:
                continue

            if _ticker_only_substring_of_word(headline, ticker, symbol):
                logger.debug("Discarded article: %s", headline)
                continue

            cat_tag_match = _matches_category_or_tags(item, ticker, symbol)
            regex_match = _matches_symbol_regex(headline, ticker, symbol) or _matches_symbol_regex(
                summary, ticker, symbol
            )

            # Discard any lowercase or ordinary prose matches for all assets
            has_prose_match = bool(
                re.search(rf"(?:^|\W){re.escape(ticker)}(?:\W|$)", headline, re.IGNORECASE)
                or re.search(rf"(?:^|\W){re.escape(ticker)}(?:\W|$)", summary, re.IGNORECASE)
            )
            if has_prose_match and not regex_match:
                logger.debug("Discarded article: %s", headline)
                continue

            # Do not keep articles where categories/tags explicitly matched other unrelated assets unless regex matches
            if cat_tag_match is False and not regex_match:
                logger.debug("Discarded article: %s", headline)
                continue

            items.append(
                {
                    "age_hours": age_hours,
                    "source": source,
                    "headline": headline,
                    "summary": summary,
                }
            )
            if len(items) >= limit:
                break

        return items

    except NewsFetchError:
        raise
    except Exception as exc:
        logger.warning(
            "Failed to fetch recent news for %s (%s): %s",
            symbol,
            type(exc).__name__,
            exc,
        )
        raise NewsFetchError(f"Failed to fetch recent news for {symbol}: {exc}") from exc


async def fetch_recent_news(
    symbol: str,
    limit: int = 10,
    *,
    timeout: float = DEFAULT_NEWS_TIMEOUT,
    async_client: httpx.AsyncClient | None = None,
    raw: bool = False,
    now: datetime | None = None,
) -> Any:
    """Fetch recent news headlines and summaries for a symbol.

    If raw=True, returns list[dict[str, Any]].
    Otherwise, returns a formatted <news_data> XML block string.
    On failure or empty feed, returns empty XML block '<news_data></news_data>' (or empty list).
    """
    items = await fetch_recent_news_items(
        symbol=symbol,
        limit=limit,
        timeout=timeout,
        async_client=async_client,
        now=now,
    )
    if raw:
        return items
    return format_news_xml(items)


__all__ = [
    "CRYPTOCOMPARE_NEWS_BASE_URL",
    "DEFAULT_NEWS_TIMEOUT",
    "NO_NEWS_FALLBACK",
    "NewsFetchError",
    "fetch_recent_news",
    "fetch_recent_news_items",
    "format_news_xml",
]
