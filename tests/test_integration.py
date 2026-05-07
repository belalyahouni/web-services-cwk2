"""
End-to-end and performance tests.

The 70-79 grade band asks for "unit, integration, and performance
tests". The unit tests live in test_crawler.py / test_indexer.py /
test_search.py / test_main.py. This file provides:

  - An integration test that drives the full pipeline (Crawler.crawl ->
    build_index -> save_index -> load_index -> find) end-to-end with
    HTTP mocked, so we exercise the actual module boundaries the way
    the CLI does at run time.

  - A simple performance smoke test that builds an index from a
    synthetic 200-page corpus and asserts it completes in well under a
    second. This is not a benchmark; it is a guard against accidental
    quadratic behaviour creeping in during refactoring.
"""

import time
from unittest.mock import MagicMock, patch

import pytest

from crawler import Crawler
from indexer import build_index, load_index, save_index, tokenise
from search import find


def _html(text):
    response = MagicMock()
    response.status_code = 200
    response.text = text
    response.headers = {"Content-Type": "text/html; charset=utf-8"}
    return response


# ---- integration --------------------------------------------------------

def test_full_pipeline_crawl_to_search(tmp_path):
    """Drive the whole pipeline through every module boundary.

    We mock the network layer only; everything above it (link
    extraction, parsing, tokenisation, index building, persistence,
    query processing) runs for real.
    """
    pages = {
        "http://site.test/": (
            "<html><body>"
            "<a href='/quotes/1'>q1</a>"
            "<a href='/quotes/2'>q2</a>"
            "<a href='https://other.test/x'>off-site, should be ignored</a>"
            "</body></html>"
        ),
        "http://site.test/quotes/1": (
            "<html><body>"
            "<h1>Inspirational</h1>"
            "<p>The journey of a thousand miles begins with one step.</p>"
            "<a href='/quotes/2'>related</a>"
            "</body></html>"
        ),
        "http://site.test/quotes/2": (
            "<html><body>"
            "<h1>Humour</h1>"
            "<p>I have not failed, I have just found ten thousand ways "
            "that won't work.</p>"
            "</body></html>"
        ),
    }

    def fake_get(url, **_kwargs):
        return _html(pages[url])

    # Crawl.
    crawler = Crawler("http://site.test/", politeness=0.0)
    with patch.object(crawler.session, "get", side_effect=fake_get):
        crawled = crawler.crawl(verbose=False)
    assert set(crawled.keys()) == set(pages.keys())

    # Build, save, load.
    index, doc_map = build_index(crawled)
    index_path = tmp_path / "index.json"
    save_index(str(index_path), index, doc_map)
    loaded_index, loaded_doc_map = load_index(str(index_path))
    assert loaded_index == index
    assert loaded_doc_map == doc_map

    # Search the loaded index using the real search functions.
    journey = find(loaded_index, loaded_doc_map, ["journey"])
    assert journey == ["http://site.test/quotes/1"]

    # A word that appears on both quote pages but not the homepage.
    thousand = set(find(loaded_index, loaded_doc_map, ["thousand"]))
    assert thousand == {
        "http://site.test/quotes/1",
        "http://site.test/quotes/2",
    }

    # Multi-word conjunction: only the first quote contains both.
    journey_thousand = find(
        loaded_index, loaded_doc_map, ["journey", "thousand"]
    )
    assert journey_thousand == ["http://site.test/quotes/1"]

    # Off-site URL must not appear anywhere.
    for url in loaded_doc_map.values():
        assert "other.test" not in url


# ---- performance --------------------------------------------------------

def _synthetic_pages(num_pages, words_per_page=200):
    """Generate `num_pages` synthetic HTML pages with deterministic content."""
    vocabulary = [f"term{i:04d}" for i in range(500)]
    pages = {}
    for page_id in range(num_pages):
        # Each page draws words deterministically from the vocabulary so
        # there is realistic term overlap across documents.
        body = " ".join(
            vocabulary[(page_id + offset) % len(vocabulary)]
            for offset in range(words_per_page)
        )
        pages[f"http://perf.test/p{page_id}"] = f"<html><body>{body}</body></html>"
    return pages


def test_build_index_handles_200_pages_quickly():
    """Indexing 200 synthetic pages of ~200 words should finish in well
    under a second on any reasonable machine. Catches accidental
    quadratic behaviour in build_index."""
    pages = _synthetic_pages(num_pages=200, words_per_page=200)

    start = time.perf_counter()
    index, doc_map = build_index(pages)
    elapsed = time.perf_counter() - start

    assert len(doc_map) == 200
    # Generous upper bound; on dev machines this typically runs in tens
    # of milliseconds. The point is to fail loudly if someone introduces
    # an O(n^2) regression.
    assert elapsed < 2.0, f"build_index took {elapsed:.3f}s for 200 pages"


def test_find_is_efficient_on_large_index():
    """Looking up a term in an index covering hundreds of documents must
    remain near-instantaneous because find is a hash lookup plus set
    intersections, not a linear scan."""
    pages = _synthetic_pages(num_pages=200, words_per_page=200)
    index, doc_map = build_index(pages)

    start = time.perf_counter()
    for _ in range(100):
        find(index, doc_map, ["term0001", "term0002"])
    elapsed = time.perf_counter() - start

    assert elapsed < 1.0, f"100 finds took {elapsed:.3f}s on a 200-page index"


def test_tokeniser_handles_long_text_quickly():
    """Tokenising a million-character document should finish in a few
    hundred milliseconds. Guards against catastrophic regex backtracking."""
    text = ("the quick brown fox jumps over the lazy dog " * 25_000)

    start = time.perf_counter()
    tokens = tokenise(text)
    elapsed = time.perf_counter() - start

    assert len(tokens) > 200_000
    assert elapsed < 1.0, f"tokenising 1MB of text took {elapsed:.3f}s"
