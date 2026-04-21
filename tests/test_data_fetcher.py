"""
Tests for indicator/data_fetcher.py — API fetching with retry, cache, and validation.

All tests mock requests.get to avoid real API calls.
"""
from __future__ import annotations

import json
import time
from unittest.mock import patch, MagicMock, PropertyMock

import numpy as np
import pandas as pd
import pytest
import requests


# ---------------------------------------------------------------------------
# Timestamp auto-detection
# ---------------------------------------------------------------------------

class TestTimestampAutoDetection:
    """Verify _cg_fetch auto-detects seconds vs milliseconds timestamps."""

    @patch("indicator.data_fetcher._retry_request")
    def test_13_digit_parsed_as_milliseconds(self, mock_request):
        """13-digit timestamps (e.g., 1775998800000) should be parsed as ms."""
        from indicator.data_fetcher import _cg_fetch

        ts_ms = 1775998800000  # 2026-04-15 ~UTC
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "code": "0",
            "data": [
                {"t": ts_ms, "close": 67000},
                {"t": ts_ms + 3600000, "close": 67100},
            ],
        }
        mock_request.return_value = mock_resp

        df = _cg_fetch("/test/path")
        assert not df.empty
        # Verify the timestamp was correctly parsed (should be 2026, not 1970)
        assert df.index[0].year >= 2026

    @patch("indicator.data_fetcher._retry_request")
    def test_10_digit_parsed_as_seconds(self, mock_request):
        """10-digit timestamps (e.g., 1775998800) should be parsed as seconds."""
        from indicator.data_fetcher import _cg_fetch

        ts_s = 1775998800  # 10-digit
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "code": "0",
            "data": [
                {"t": ts_s, "close": 67000},
                {"t": ts_s + 3600, "close": 67100},
            ],
        }
        mock_request.return_value = mock_resp

        df = _cg_fetch("/test/path")
        assert not df.empty
        assert df.index[0].year >= 2026

    @patch("indicator.data_fetcher._retry_request")
    def test_time_column_alternatives(self, mock_request):
        """Should handle 'time', 't', and 'createTime' column names."""
        from indicator.data_fetcher import _cg_fetch

        ts = 1775998800000
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "code": "0",
            "data": [{"time": ts, "value": 1.0}],
        }
        mock_request.return_value = mock_resp

        df = _cg_fetch("/test/path")
        assert not df.empty
        assert df.index[0].year >= 2026


# ---------------------------------------------------------------------------
# API failure fallback
# ---------------------------------------------------------------------------

class TestAPIFailureFallback:
    """Verify fallback to cached data when API calls fail."""

    @patch("indicator.data_fetcher._load_cache")
    @patch("indicator.data_fetcher._retry_request")
    def test_binance_klines_uses_cache_on_failure(self, mock_request, mock_cache):
        """When Binance API fails, should return cached data."""
        from indicator.data_fetcher import fetch_binance_klines

        mock_request.side_effect = requests.RequestException("Connection refused")
        cached_df = pd.DataFrame({"close": [67000, 67100]})
        mock_cache.return_value = cached_df

        result = fetch_binance_klines()
        mock_cache.assert_called_with("binance_klines")
        assert len(result) == 2

    @patch("indicator.data_fetcher._load_cache")
    @patch("indicator.data_fetcher._retry_request")
    def test_binance_klines_fallback_on_empty_response(self, mock_request, mock_cache):
        """Empty response from Binance should trigger cache fallback."""
        from indicator.data_fetcher import fetch_binance_klines

        mock_resp = MagicMock()
        mock_resp.json.return_value = []
        mock_request.return_value = mock_resp

        cached_df = pd.DataFrame({"close": [67000]})
        mock_cache.return_value = cached_df

        result = fetch_binance_klines()
        mock_cache.assert_called_with("binance_klines")

    @patch("indicator.data_fetcher._load_cache")
    @patch("indicator.data_fetcher._retry_request")
    def test_coinglass_uses_cache_on_failure(self, mock_request, mock_cache):
        """When CG API fails, individual endpoints should use cached data."""
        from indicator.data_fetcher import _cg_fetch

        mock_request.side_effect = requests.RequestException("Timeout")
        cached_df = pd.DataFrame({"close": [100.0]})
        mock_cache.return_value = cached_df

        # _cg_fetch raises, caller (fetch_coinglass) catches and uses cache
        with pytest.raises(requests.RequestException):
            _cg_fetch("/test/path")


# ---------------------------------------------------------------------------
# Response validation
# ---------------------------------------------------------------------------

class TestResponseValidation:
    """Verify that malformed responses are handled gracefully."""

    @patch("indicator.data_fetcher._retry_request")
    def test_cg_non_zero_code_returns_empty(self, mock_request):
        """CG response with code != '0' should return empty DataFrame."""
        from indicator.data_fetcher import _cg_fetch

        mock_resp = MagicMock()
        mock_resp.json.return_value = {"code": "40001", "msg": "Invalid API key"}
        mock_request.return_value = mock_resp

        df = _cg_fetch("/test/path")
        assert df.empty

    @patch("indicator.data_fetcher._retry_request")
    def test_cg_no_data_field_returns_empty(self, mock_request):
        """CG response with missing data field should return empty DataFrame."""
        from indicator.data_fetcher import _cg_fetch

        mock_resp = MagicMock()
        mock_resp.json.return_value = {"code": "0", "data": None}
        mock_request.return_value = mock_resp

        df = _cg_fetch("/test/path")
        assert df.empty

    @patch("indicator.data_fetcher._retry_request")
    def test_cg_empty_data_list(self, mock_request):
        """CG response with empty data list should return empty DataFrame."""
        from indicator.data_fetcher import _cg_fetch

        mock_resp = MagicMock()
        mock_resp.json.return_value = {"code": "0", "data": []}
        mock_request.return_value = mock_resp

        df = _cg_fetch("/test/path")
        assert df.empty

    @patch("indicator.data_fetcher._load_cache")
    @patch("indicator.data_fetcher._retry_request")
    def test_binance_negative_prices_trigger_cache(self, mock_request, mock_cache):
        """Binance klines with negative prices should fall back to cache."""
        from indicator.data_fetcher import fetch_binance_klines

        # Build a response with negative close price
        rows = []
        for i in range(10):
            rows.append([
                1775998800000 + i * 3600000,  # ts_open
                "67000", "67100", "66900", "-1",  # OHLC (invalid close)
                "1000", 1776002400000, "67000000",  # vol, close_time, quote_vol
                10000, "500", "33500000", "0",  # trade_count, taker_buy, taker_quote, ignore
            ])

        mock_resp = MagicMock()
        mock_resp.json.return_value = rows
        mock_request.return_value = mock_resp

        cached = pd.DataFrame({"close": [67000.0]})
        mock_cache.return_value = cached

        result = fetch_binance_klines()
        # Should have fallen back to cache due to invalid prices
        mock_cache.assert_called()


# ---------------------------------------------------------------------------
# Retry mechanism
# ---------------------------------------------------------------------------

class TestRetryMechanism:
    """Verify exponential backoff retry logic."""

    @patch("indicator.data_fetcher.time.sleep")
    def test_retry_on_failure_then_success(self, mock_sleep):
        """Should retry on failure and return on success."""
        from indicator.data_fetcher import _retry_request

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()

        with patch("indicator.data_fetcher.requests.request") as mock_req:
            mock_req.side_effect = [
                requests.ConnectionError("fail"),
                mock_resp,
            ]
            result = _retry_request("GET", "http://test.com")

        assert result == mock_resp
        # Should have slept once (between attempt 1 and 2)
        mock_sleep.assert_called_once()

    @patch("indicator.data_fetcher.time.sleep")
    def test_retry_exhaustion_raises(self, mock_sleep):
        """After MAX_RETRIES failures, should raise the last exception."""
        from indicator.data_fetcher import _retry_request, MAX_RETRIES

        with patch("indicator.data_fetcher.requests.request") as mock_req:
            mock_req.side_effect = requests.ConnectionError("persistent failure")

            with pytest.raises(requests.ConnectionError):
                _retry_request("GET", "http://test.com")

        # Should have retried MAX_RETRIES times
        assert mock_req.call_count == MAX_RETRIES

    @patch("indicator.data_fetcher.time.sleep")
    def test_exponential_backoff_delays(self, mock_sleep):
        """Delay between retries should double each time."""
        from indicator.data_fetcher import _retry_request, RETRY_BASE_DELAY

        with patch("indicator.data_fetcher.requests.request") as mock_req:
            mock_req.side_effect = requests.ConnectionError("fail")

            with pytest.raises(requests.ConnectionError):
                _retry_request("GET", "http://test.com")

        # Check delay pattern: base, base*2 (for 3 retries, 2 sleeps)
        delays = [c[0][0] for c in mock_sleep.call_args_list]
        assert delays[0] == RETRY_BASE_DELAY
        assert delays[1] == RETRY_BASE_DELAY * 2


# ---------------------------------------------------------------------------
# Binance depth
# ---------------------------------------------------------------------------

class TestBinanceDepth:
    """Tests for fetch_binance_depth()."""

    @patch("indicator.data_fetcher._retry_request")
    def test_depth_imbalance_calculation(self, mock_request):
        """Depth imbalance = (bid - ask) / (bid + ask)."""
        from indicator.data_fetcher import fetch_binance_depth

        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "bids": [["67000", "10"], ["66990", "5"]],
            "asks": [["67010", "8"], ["67020", "4"]],
        }
        mock_request.return_value = mock_resp

        result = fetch_binance_depth()
        assert "depth_imbalance" in result
        assert "bid_depth_usd" in result
        assert "spread_bps" in result

        # Bid depth = 67000*10 + 66990*5 = 670000 + 334950 = 1004950
        # Ask depth = 67010*8 + 67020*4 = 536080 + 268080 = 804160
        expected_imb = (1004950 - 804160) / (1004950 + 804160)
        assert result["depth_imbalance"] == pytest.approx(expected_imb, rel=1e-4)

    @patch("indicator.data_fetcher._retry_request")
    def test_empty_depth_returns_empty(self, mock_request):
        """Empty order book should return empty dict."""
        from indicator.data_fetcher import fetch_binance_depth

        mock_resp = MagicMock()
        mock_resp.json.return_value = {"bids": [], "asks": []}
        mock_request.return_value = mock_resp

        result = fetch_binance_depth()
        assert result == {}


# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------

class TestRateLimiting:
    """Verify delay between Coinglass API calls."""

    @patch("indicator.data_fetcher.time.sleep")
    @patch("indicator.data_fetcher._cg_fetch")
    @patch("indicator.data_fetcher._save_cache")
    @patch("indicator.data_fetcher._load_cache")
    def test_coinglass_sleeps_between_calls(self, mock_load, mock_save,
                                             mock_fetch, mock_sleep):
        """fetch_coinglass() should sleep 1s between endpoint calls."""
        from indicator.data_fetcher import fetch_coinglass

        mock_fetch.return_value = pd.DataFrame()
        mock_load.return_value = pd.DataFrame()

        with patch.dict("os.environ", {"COINGLASS_API_KEY": "test_key"}):
            fetch_coinglass()

        # Should have called sleep(1) between each endpoint
        sleep_calls = [c for c in mock_sleep.call_args_list if c[0][0] == 1]
        assert len(sleep_calls) > 0, "Should sleep between CG endpoint calls"


# ---------------------------------------------------------------------------
# Deribit
# ---------------------------------------------------------------------------

class TestDeribitDVOL:
    """Tests for fetch_deribit_dvol()."""

    @patch("indicator.data_fetcher._retry_request")
    def test_dvol_parsing(self, mock_request):
        """Should correctly parse DVOL OHLC candle data."""
        from indicator.data_fetcher import fetch_deribit_dvol

        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "result": {
                "data": [
                    [1775998800000, 55.0, 58.0, 53.0, 56.5],
                ]
            }
        }
        mock_request.return_value = mock_resp

        result = fetch_deribit_dvol()
        assert result["dvol_value"] == 56.5
        assert result["dvol_open"] == 55.0
        assert result["dvol_high"] == 58.0
        assert result["dvol_low"] == 53.0
        assert result["dvol_change"] == pytest.approx(1.5)

    @patch("indicator.data_fetcher._retry_request")
    def test_dvol_empty_data(self, mock_request):
        """Empty DVOL data should return empty dict."""
        from indicator.data_fetcher import fetch_deribit_dvol

        mock_resp = MagicMock()
        mock_resp.json.return_value = {"result": {"data": []}}
        mock_request.return_value = mock_resp

        result = fetch_deribit_dvol()
        assert result == {}

    @patch("indicator.data_fetcher._retry_request")
    def test_dvol_api_failure(self, mock_request):
        """API failure should return empty dict, not crash."""
        from indicator.data_fetcher import fetch_deribit_dvol

        mock_request.side_effect = requests.RequestException("timeout")

        result = fetch_deribit_dvol()
        assert result == {}


# ---------------------------------------------------------------------------
# Cache mechanism
# ---------------------------------------------------------------------------

class TestCacheMechanism:
    """Tests for _save_cache and _load_cache."""

    @patch("indicator.data_fetcher.CACHE_DIR")
    def test_load_cache_missing_file(self, mock_dir, tmp_path):
        """Loading a non-existent cache should return empty DataFrame."""
        from indicator.data_fetcher import _load_cache

        mock_dir.__truediv__ = lambda self, x: tmp_path / x
        mock_dir.mkdir = MagicMock()

        result = _load_cache("nonexistent_endpoint")
        assert result.empty
