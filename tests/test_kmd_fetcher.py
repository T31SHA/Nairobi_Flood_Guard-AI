"""KMD fetcher tests (offline / mocked)."""

from unittest.mock import patch

from Utils.kmd_fetcher import fetch_kmd_advisory


def test_linkout_fallback_when_no_feed():
    with patch("Utils.kmd_fetcher.requests.get") as mock_get:
        mock_get.return_value.status_code = 404
        mock_get.return_value.text = "<html></html>"
        result = fetch_kmd_advisory(force=True)
    assert result["mode"] in ("linkout", "scraped")
    assert "urls" in result
    assert "weather_warnings" in result["urls"]
