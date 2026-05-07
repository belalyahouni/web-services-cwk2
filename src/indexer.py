"""
Inverted index construction for the COMP3011 search engine tool.

Two-pass parsing follows Lecture 11 (Parsing and Tokenisation):
  Pass 1 - parse the HTML markup with BeautifulSoup, discarding
           non-content tags such as <script> and <style>.
  Pass 2 - tokenise the visible text into lowercase words.

The inverted index follows Lecture 12 (Indexing): each term maps to a
posting list. Each posting records the document the term appears in,
the term's frequency in that document, and every position at which it
appears. Storing positions lets future query processing support phrase
queries (L12 slide 15 "Positions").

Per the project's tokenisation decision (and L11's caveat about queries
like "to be or not to be") all words are kept; no stopword removal and
no stemming are applied.

Persistence uses a single JSON file holding both the inverted index and
the document-id -> URL map (L12 slide 25 "Inverted Files (continued)" -
inverted lists are stored together in one file rather than fragmented).

This module was developed with AI assistance (Claude); all design
choices were reviewed and adjusted by the author.
"""

import json
import re

from bs4 import BeautifulSoup


# Matches runs of alphanumeric characters. Lowercasing happens before
# the regex runs, so [a-z0-9]+ is sufficient. This matches the simple
# "alphanumeric tokenisation" from L11 slide 7 - imperfect on hyphenated
# words and contractions but adequate for quotes.toscrape.com.
TOKEN_PATTERN = re.compile(r"[a-z0-9]+")

# Tags whose contents are not meaningful page text.
NON_CONTENT_TAGS = ("script", "style", "noscript")


def extract_text(html):
    """Return the visible text of an HTML document (L11 pass 1)."""
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(NON_CONTENT_TAGS):
        tag.decompose()
    return soup.get_text(separator=" ")


def tokenise(text):
    """Lowercase the text and split it into alphanumeric tokens (L11 pass 2).

    All tokens are kept (including stopwords and numbers). Positions are
    implicit in the returned list order; callers use enumerate() to
    record them.
    """
    return TOKEN_PATTERN.findall(text.lower())


def build_index(pages):
    """Build an inverted index from a {url: html} mapping.

    Returns:
        index:   {term: {doc_id: {"frequency": int, "positions": [int]}}}
        doc_map: {doc_id: url}

    doc_id is a string so the structure round-trips through JSON without
    type coercion.
    """
    doc_map = {}
    index = {}
    for ordinal, (url, html) in enumerate(pages.items()):
        doc_id = str(ordinal)
        doc_map[doc_id] = url
        text = extract_text(html)
        for position, term in enumerate(tokenise(text)):
            postings = index.setdefault(term, {})
            entry = postings.setdefault(doc_id, {"frequency": 0, "positions": []})
            entry["frequency"] += 1
            entry["positions"].append(position)
    return index, doc_map


def save_index(path, index, doc_map):
    """Persist the index and document map to a single JSON file."""
    with open(path, "w", encoding="utf-8") as handle:
        json.dump({"doc_map": doc_map, "index": index}, handle)


def load_index(path):
    """Load an index previously written by `save_index`.

    Returns (index, doc_map).
    """
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    return data["index"], data["doc_map"]
