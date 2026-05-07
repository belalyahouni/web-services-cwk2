"""
Tests for the crawler module. The network is mocked so the test suite
runs offline and instantly. Coverage focuses on the behaviours required
by the brief and motivated in L9: politeness window, host restriction,
visited-set discipline, and graceful error handling.
"""

import time
from unittest.mock import MagicMock, patch

import pytest

from crawler import Crawler


def _html_response(text, status=200, content_type="text/html; charset=utf-8"):
    """Build a fake requests.Response."""
    response = MagicMock()
    response.status_code = status
    response.text = text
    response.headers = {"Content-Type": content_type}
    return response


def test_crawl_returns_html_for_visited_pages():
    pages_by_url = {
        "https://example.com/": "<html><body><a href='/a'>a</a></body></html>",
        "https://example.com/a": "<html><body>page a</body></html>",
    }

    def fake_get(url, **_kwargs):
        return _html_response(pages_by_url[url])

    crawler = Crawler("https://example.com/", politeness=0.0)
    with patch.object(crawler.session, "get", side_effect=fake_get):
        result = crawler.crawl(verbose=False)

    assert set(result.keys()) == set(pages_by_url.keys())


def test_only_internal_links_are_followed():
    seed = "<html><body>" \
           "<a href='/inside'>i</a>" \
           "<a href='https://other.example.com/x'>o</a>" \
           "</body></html>"
    inside = "<html><body>inside</body></html>"

    def fake_get(url, **_kwargs):
        if url == "https://example.com/":
            return _html_response(seed)
        if url == "https://example.com/inside":
            return _html_response(inside)
        raise AssertionError(f"crawler should not request {url}")

    crawler = Crawler("https://example.com/", politeness=0.0)
    with patch.object(crawler.session, "get", side_effect=fake_get):
        result = crawler.crawl(verbose=False)

    assert "https://example.com/inside" in result
    assert "https://other.example.com/x" not in result


def test_visited_pages_are_not_refetched():
    seed = "<html><a href='/a'>a</a><a href='/a'>a-again</a></html>"
    a_html = "<html><a href='/'>back</a></html>"

    call_counts = {"https://example.com/": 0, "https://example.com/a": 0}

    def fake_get(url, **_kwargs):
        call_counts[url] = call_counts.get(url, 0) + 1
        if url == "https://example.com/":
            return _html_response(seed)
        if url == "https://example.com/a":
            return _html_response(a_html)
        return _html_response("", status=404)

    crawler = Crawler("https://example.com/", politeness=0.0)
    with patch.object(crawler.session, "get", side_effect=fake_get):
        crawler.crawl(verbose=False)

    assert call_counts["https://example.com/"] == 1
    assert call_counts["https://example.com/a"] == 1


def test_url_fragments_are_normalised():
    seed = "<html><a href='/a#top'>a</a><a href='/a#bottom'>a2</a></html>"
    a_html = "<html>a</html>"
    seen = []

    def fake_get(url, **_kwargs):
        seen.append(url)
        if url == "https://example.com/":
            return _html_response(seed)
        return _html_response(a_html)

    crawler = Crawler("https://example.com/", politeness=0.0)
    with patch.object(crawler.session, "get", side_effect=fake_get):
        crawler.crawl(verbose=False)

    # Both /a#top and /a#bottom should collapse to /a, so /a is fetched once.
    assert seen.count("https://example.com/a") == 1


def test_politeness_window_is_respected():
    """The crawler must wait at least `politeness` seconds between requests."""
    seed = "<html><a href='/a'>a</a><a href='/b'>b</a></html>"
    bodies = {
        "https://example.com/": seed,
        "https://example.com/a": "<html>a</html>",
        "https://example.com/b": "<html>b</html>",
    }
    timestamps = []

    def fake_get(url, **_kwargs):
        timestamps.append(time.monotonic())
        return _html_response(bodies[url])

    politeness = 0.2  # short for test speed; behaviour is identical to 6s.
    crawler = Crawler("https://example.com/", politeness=politeness)
    with patch.object(crawler.session, "get", side_effect=fake_get):
        crawler.crawl(verbose=False)

    assert len(timestamps) == 3
    for earlier, later in zip(timestamps, timestamps[1:]):
        assert later - earlier >= politeness - 0.01  # small clock tolerance


def test_non_200_response_is_skipped():
    """A 404 page must not appear in the results and must not crash."""
    seed = "<html><a href='/missing'>m</a></html>"

    def fake_get(url, **_kwargs):
        if url == "https://example.com/":
            return _html_response(seed)
        return _html_response("", status=404)

    crawler = Crawler("https://example.com/", politeness=0.0)
    with patch.object(crawler.session, "get", side_effect=fake_get):
        result = crawler.crawl(verbose=False)

    assert "https://example.com/missing" not in result
    assert "https://example.com/" in result


def test_network_exception_is_handled_gracefully():
    """Network errors should not crash the crawl."""
    import requests

    seed = "<html><a href='/dead'>d</a></html>"

    def fake_get(url, **_kwargs):
        if url == "https://example.com/":
            return _html_response(seed)
        raise requests.ConnectionError("simulated dropped connection")

    crawler = Crawler("https://example.com/", politeness=0.0)
    with patch.object(crawler.session, "get", side_effect=fake_get):
        result = crawler.crawl(verbose=False)

    assert "https://example.com/dead" not in result


def test_max_pages_limit_is_honoured():
    """When max_pages is set, the crawler stops at that limit."""
    seed = "<html>" + "".join(f"<a href='/p{i}'>p{i}</a>" for i in range(5)) + "</html>"

    def fake_get(url, **_kwargs):
        if url == "https://example.com/":
            return _html_response(seed)
        return _html_response("<html>page</html>")

    crawler = Crawler("https://example.com/", politeness=0.0)
    with patch.object(crawler.session, "get", side_effect=fake_get):
        result = crawler.crawl(max_pages=3, verbose=False)

    assert len(result) == 3


def test_non_html_content_type_is_skipped():
    """Links pointing at non-HTML resources (PDFs, images) should not be
    treated as pages."""
    seed = "<html><a href='/file.pdf'>pdf</a></html>"

    def fake_get(url, **_kwargs):
        if url == "https://example.com/":
            return _html_response(seed)
        # PDF response: content-type is not text/html.
        return _html_response(
            "%PDF-1.4 ...", content_type="application/pdf"
        )

    crawler = Crawler("https://example.com/", politeness=0.0)
    with patch.object(crawler.session, "get", side_effect=fake_get):
        result = crawler.crawl(verbose=False)

    assert "https://example.com/file.pdf" not in result
    assert "https://example.com/" in result


def test_non_http_schemes_are_filtered_out():
    """Anchors with mailto:, javascript:, tel: schemes must not be crawled."""
    seed = (
        "<html>"
        "<a href='mailto:hi@example.com'>email</a>"
        "<a href='javascript:void(0)'>js</a>"
        "<a href='tel:+1234'>phone</a>"
        "<a href='/real'>real</a>"
        "</html>"
    )
    real = "<html>real page</html>"
    seen = []

    def fake_get(url, **_kwargs):
        seen.append(url)
        if url == "https://example.com/":
            return _html_response(seed)
        if url == "https://example.com/real":
            return _html_response(real)
        raise AssertionError(f"crawler should not request {url}")

    crawler = Crawler("https://example.com/", politeness=0.0)
    with patch.object(crawler.session, "get", side_effect=fake_get):
        result = crawler.crawl(verbose=False)

    assert set(result.keys()) == {
        "https://example.com/",
        "https://example.com/real",
    }
    # The non-http schemes were never requested.
    for url in seen:
        assert url.startswith("https://example.com/")


def test_relative_urls_are_resolved_against_the_base():
    """Anchors using relative paths must resolve to absolute internal URLs."""
    seed = "<html><a href='./sub/page'>p</a><a href='../top'>t</a></html>"
    sub = "<html>sub page</html>"
    top = "<html>top page</html>"

    def fake_get(url, **_kwargs):
        return _html_response({
            "https://example.com/start/": seed,
            "https://example.com/start/sub/page": sub,
            "https://example.com/top": top,
        }.get(url, ""))

    crawler = Crawler("https://example.com/start/", politeness=0.0)
    with patch.object(crawler.session, "get", side_effect=fake_get):
        result = crawler.crawl(verbose=False)

    assert "https://example.com/start/sub/page" in result
    assert "https://example.com/top" in result


def test_verbose_mode_prints_progress(capsys):
    """When verbose=True the crawler prints a fetching line per page."""
    pages_by_url = {"https://example.com/": "<html>only</html>"}

    def fake_get(url, **_kwargs):
        return _html_response(pages_by_url[url])

    crawler = Crawler("https://example.com/", politeness=0.0)
    with patch.object(crawler.session, "get", side_effect=fake_get):
        crawler.crawl(verbose=True)

    captured = capsys.readouterr()
    assert "fetching https://example.com/" in captured.out
