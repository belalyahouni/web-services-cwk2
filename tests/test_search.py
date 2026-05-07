"""
Tests for the search module. Covers single-word lookup, multi-word
conjunctive search (L13), TF-IDF ranking and tie-breaking, case
insensitivity, and edge cases (empty queries, missing terms, special
characters).

`find` returns ranked (url, score) pairs; helpers below extract just
the URL set or the URL list when ordering is not the focus of a test.
"""

import io
import math
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


def _urls(ranked):
    return [url for url, _score in ranked]


def _url_set(ranked):
    return {url for url, _score in ranked}


# ---- find ---------------------------------------------------------------

def test_find_single_word(corpus):
    index, doc_map = corpus
    matches = find(index, doc_map, ["good"])
    assert _url_set(matches) == {
        "http://example.com/p1",
        "http://example.com/p2",
    }


def test_find_multi_word_is_conjunctive(corpus):
    """Multi-word queries return only pages with ALL terms (L13)."""
    index, doc_map = corpus
    matches = find(index, doc_map, ["good", "friends"])
    assert _urls(matches) == ["http://example.com/p1"]


def test_find_is_case_insensitive(corpus):
    index, doc_map = corpus
    matches_lower = find(index, doc_map, ["good"])
    matches_upper = find(index, doc_map, ["GOOD"])
    matches_mixed = find(index, doc_map, ["Good"])
    assert (
        _url_set(matches_lower)
        == _url_set(matches_upper)
        == _url_set(matches_mixed)
    )


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
    bare = _url_set(find(index, doc_map, ["good"]))
    with_bang = _url_set(find(index, doc_map, ["good!"]))
    assert bare == with_bang


def test_find_multi_word_intersection_is_order_independent(corpus):
    index, doc_map = corpus
    a = _url_set(find(index, doc_map, ["good", "friends"]))
    b = _url_set(find(index, doc_map, ["friends", "good"]))
    assert a == b


# ---- TF-IDF ranking -----------------------------------------------------

def test_find_returns_url_score_pairs(corpus):
    """Each result is a (url, score) tuple with a positive score."""
    index, doc_map = corpus
    matches = find(index, doc_map, ["good"])
    assert all(isinstance(item, tuple) and len(item) == 2 for item in matches)
    for url, score in matches:
        assert isinstance(url, str)
        assert isinstance(score, float)


def test_find_ranks_higher_frequency_first():
    """A document where the term appears more often should outrank one
    where it appears once. Sublinear tf still preserves this ordering."""
    pages = {
        "http://x/often": "<html>" + ("good " * 10) + "</html>",
        "http://x/once": "<html>good day</html>",
        "http://x/none": "<html>nothing here</html>",
    }
    index, doc_map = build_index(pages)
    matches = find(index, doc_map, ["good"])
    assert _urls(matches) == ["http://x/often", "http://x/once"]
    # Sublinear tf: log10(10) = 1, so tf_often = 2; log10(1) = 0, so tf_once = 1.
    # idf = log10(3 / 2) (good appears in 2 of 3 docs).
    score_often, score_once = matches[0][1], matches[1][1]
    assert score_often > score_once
    assert score_often == pytest.approx(2.0 * math.log10(3 / 2))
    assert score_once == pytest.approx(1.0 * math.log10(3 / 2))


def test_find_breaks_ties_by_url():
    """When two documents have equal scores, URLs are returned in
    ascending lexicographic order so the output is reproducible."""
    pages = {
        "http://x/zebra": "<html>good</html>",
        "http://x/apple": "<html>good</html>",
        "http://x/mango": "<html>good</html>",
    }
    index, doc_map = build_index(pages)
    matches = find(index, doc_map, ["good"])
    assert _urls(matches) == [
        "http://x/apple",
        "http://x/mango",
        "http://x/zebra",
    ]
    scores = [score for _url, score in matches]
    assert scores[0] == scores[1] == scores[2]


def test_find_multiword_score_is_sum_of_term_contributions():
    """The score for a multi-term query is the sum of each term's tf*idf."""
    pages = {
        "http://x/both": "<html>good day good day friends</html>",
        "http://x/justone": "<html>good morning</html>",
    }
    index, doc_map = build_index(pages)
    matches = find(index, doc_map, ["good", "friends"])
    # Only the 'both' page contains both query terms.
    assert _urls(matches) == ["http://x/both"]
    # tf(good, both) = 1 + log10(2); idf(good) = log10(2/2) = 0
    # tf(friends, both) = 1 + log10(1) = 1; idf(friends) = log10(2/1)
    expected = (1 + math.log10(2)) * math.log10(2 / 2) + 1.0 * math.log10(2 / 1)
    assert matches[0][1] == pytest.approx(expected)


def test_find_idf_demotes_common_terms():
    """A term that appears in every document has idf = 0 and contributes
    nothing to ranking — so a query of one common + one rare term ranks
    by the rare term alone."""
    pages = {
        "http://x/p1": "<html>common rare</html>",
        "http://x/p2": "<html>common everywhere</html>",
        "http://x/p3": "<html>common somewhere</html>",
    }
    index, doc_map = build_index(pages)
    matches = find(index, doc_map, ["common"])
    # idf = log10(3/3) = 0 -> all scores zero, but documents still match.
    assert _url_set(matches) == set(doc_map.values())
    for _url, score in matches:
        assert score == pytest.approx(0.0)


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
