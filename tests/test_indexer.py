"""
Tests for the indexer module: tokenisation, HTML extraction, inverted
index construction, and JSON round-tripping. Edge cases follow L11
(case folding, punctuation, numbers) and L12 (frequencies and positions
in postings).
"""

import json
import os

import pytest

from indexer import (
    build_index,
    extract_text,
    load_index,
    save_index,
    tokenise,
)


# ---- tokenise -----------------------------------------------------------

def test_tokenise_lowercases_input():
    assert tokenise("Hello World") == ["hello", "world"]


def test_tokenise_strips_punctuation():
    assert tokenise("hello, world! it's fine.") == ["hello", "world", "it", "s", "fine"]


def test_tokenise_keeps_numbers():
    # L11 slide 13 notes that numbers can be important.
    assert tokenise("Nokia 3250 top 10") == ["nokia", "3250", "top", "10"]


def test_tokenise_empty_string():
    assert tokenise("") == []


def test_tokenise_keeps_stopwords():
    # Project decision: keep all words so queries like "to be or not to
    # be" still work (L11 caveat on stopword removal).
    tokens = tokenise("To be or not to be")
    assert tokens == ["to", "be", "or", "not", "to", "be"]


def test_tokenise_handles_repeated_whitespace():
    assert tokenise("a   b\t\nc") == ["a", "b", "c"]


# ---- extract_text -------------------------------------------------------

def test_extract_text_strips_script_and_style():
    html = """
    <html><head><style>body{color:red}</style></head>
    <body>
      <script>var secret = 42;</script>
      <p>visible content</p>
    </body></html>
    """
    text = extract_text(html)
    assert "secret" not in text
    assert "color" not in text
    assert "visible content" in text


def test_extract_text_returns_visible_words():
    html = "<html><body><h1>Title</h1><p>Body text here.</p></body></html>"
    tokens = tokenise(extract_text(html))
    assert "title" in tokens
    assert "body" in tokens
    assert "text" in tokens
    assert "here" in tokens


# ---- build_index --------------------------------------------------------

def _simple_pages():
    return {
        "http://example.com/a": "<html><body>fish fish chips</body></html>",
        "http://example.com/b": "<html><body>chips and salt</body></html>",
    }


def test_build_index_returns_index_and_doc_map():
    index, doc_map = build_index(_simple_pages())
    assert set(doc_map.values()) == {
        "http://example.com/a",
        "http://example.com/b",
    }
    assert "fish" in index
    assert "chips" in index


def test_build_index_records_correct_frequencies():
    index, _ = build_index(_simple_pages())
    fish_postings = index["fish"]
    # Only one document contains 'fish', and it appears twice.
    assert len(fish_postings) == 1
    only_doc = next(iter(fish_postings.values()))
    assert only_doc["frequency"] == 2


def test_build_index_records_correct_positions():
    pages = {"http://x/": "<html><body>alpha beta alpha gamma</body></html>"}
    index, _ = build_index(pages)
    only_alpha = next(iter(index["alpha"].values()))
    assert only_alpha["positions"] == [0, 2]
    only_beta = next(iter(index["beta"].values()))
    assert only_beta["positions"] == [1]
    only_gamma = next(iter(index["gamma"].values()))
    assert only_gamma["positions"] == [3]


def test_build_index_doc_ids_are_strings():
    """Doc-ids are strings so the index round-trips through JSON cleanly."""
    index, doc_map = build_index(_simple_pages())
    for doc_id in doc_map:
        assert isinstance(doc_id, str)
    for postings in index.values():
        for doc_id in postings:
            assert isinstance(doc_id, str)


def test_build_index_is_case_insensitive():
    pages = {"http://x/": "<html>Good GOOD good</html>"}
    index, _ = build_index(pages)
    assert "good" in index
    assert "Good" not in index
    assert "GOOD" not in index
    only = next(iter(index["good"].values()))
    assert only["frequency"] == 3


# ---- save_index / load_index --------------------------------------------

def test_save_and_load_round_trip(tmp_path):
    index, doc_map = build_index(_simple_pages())
    path = tmp_path / "index.json"
    save_index(str(path), index, doc_map)

    loaded_index, loaded_doc_map = load_index(str(path))
    assert loaded_index == index
    assert loaded_doc_map == doc_map


def test_save_index_writes_valid_json(tmp_path):
    index, doc_map = build_index(_simple_pages())
    path = tmp_path / "index.json"
    save_index(str(path), index, doc_map)

    # Plain json.load must succeed and the structure must contain both keys.
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    assert "index" in data
    assert "doc_map" in data


def test_save_index_to_single_file(tmp_path):
    """Per L12 (and the brief), the index lives in one file - not many."""
    index, doc_map = build_index(_simple_pages())
    path = tmp_path / "index.json"
    save_index(str(path), index, doc_map)

    files = [f for f in os.listdir(tmp_path) if not f.startswith(".")]
    assert files == ["index.json"]
