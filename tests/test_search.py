"""
Tests for the search module. Covers single-word lookup, multi-word
conjunctive search (L13), case-insensitive querying, and edge cases
(empty queries, missing terms, special characters).
"""

import io
from contextlib import redirect_stdout

import pytest

from indexer import build_index
from search import find, print_postings


# A tiny fixed corpus so the assertions are easy to read.
PAGES = {
    "http://example.com/p1": "<html>good morning friends</html>",
    "http://example.com/p2": "<html>good afternoon</html>",
    "http://example.com/p3": "<html>friends forever</html>",
    "http://example.com/p4": "<html>quiet page</html>",
}


@pytest.fixture
def corpus():
    return build_index(PAGES)


# ---- find ---------------------------------------------------------------

def test_find_single_word(corpus):
    index, doc_map = corpus
    matches = find(index, doc_map, ["good"])
    assert set(matches) == {"http://example.com/p1", "http://example.com/p2"}


def test_find_multi_word_is_conjunctive(corpus):
    """Multi-word queries return only pages with ALL terms (L13)."""
    index, doc_map = corpus
    matches = find(index, doc_map, ["good", "friends"])
    assert matches == ["http://example.com/p1"]


def test_find_is_case_insensitive(corpus):
    index, doc_map = corpus
    matches_lower = find(index, doc_map, ["good"])
    matches_upper = find(index, doc_map, ["GOOD"])
    matches_mixed = find(index, doc_map, ["Good"])
    assert set(matches_lower) == set(matches_upper) == set(matches_mixed)


def test_find_missing_term_returns_no_matches(corpus):
    index, doc_map = corpus
    assert find(index, doc_map, ["nonsense"]) == []


def test_find_missing_term_in_conjunction_returns_no_matches(corpus):
    """If even one query term is missing, the conjunction is empty."""
    index, doc_map = corpus
    assert find(index, doc_map, ["good", "nonsense"]) == []


def test_find_with_empty_query(corpus):
    index, doc_map = corpus
    assert find(index, doc_map, []) == []


def test_find_query_with_only_punctuation(corpus):
    """A query that tokenises to no real terms returns no results."""
    index, doc_map = corpus
    assert find(index, doc_map, ["!?,."]) == []


def test_find_strips_punctuation_from_query(corpus):
    """'good!' must match the same docs as 'good'."""
    index, doc_map = corpus
    bare = set(find(index, doc_map, ["good"]))
    with_bang = set(find(index, doc_map, ["good!"]))
    assert bare == with_bang


def test_find_multi_word_intersection_is_order_independent(corpus):
    index, doc_map = corpus
    a = set(find(index, doc_map, ["good", "friends"]))
    b = set(find(index, doc_map, ["friends", "good"]))
    assert a == b


# ---- print_postings -----------------------------------------------------

def test_print_postings_shows_url_and_stats(corpus):
    index, doc_map = corpus
    out = io.StringIO()
    with redirect_stdout(out):
        print_postings(index, doc_map, "good")
    text = out.getvalue()
    assert "http://example.com/p1" in text
    assert "http://example.com/p2" in text
    assert "frequency" in text
    assert "positions" in text


def test_print_postings_handles_missing_word(corpus):
    index, doc_map = corpus
    out = io.StringIO()
    with redirect_stdout(out):
        print_postings(index, doc_map, "missing")
    assert "No entry" in out.getvalue()


def test_print_postings_is_case_insensitive(corpus):
    index, doc_map = corpus
    out_lower = io.StringIO()
    out_upper = io.StringIO()
    with redirect_stdout(out_lower):
        print_postings(index, doc_map, "good")
    with redirect_stdout(out_upper):
        print_postings(index, doc_map, "GOOD")
    # Both should print the same set of URLs (frequency/positions identical too).
    assert "p1" in out_lower.getvalue() and "p2" in out_lower.getvalue()
    assert "p1" in out_upper.getvalue() and "p2" in out_upper.getvalue()


def test_print_postings_handles_empty_word(corpus):
    index, doc_map = corpus
    out = io.StringIO()
    with redirect_stdout(out):
        print_postings(index, doc_map, "")
    assert "Usage" in out.getvalue()
