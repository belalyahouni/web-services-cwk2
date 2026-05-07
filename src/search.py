"""
Query processing for the COMP3011 search engine tool.

`print_postings` shows the raw posting list for a term (the `print`
command in the brief).

`find` implements multi-word search with conjunctive processing as
described in Lecture 13 (Query Processing):

  - "By conjunctive processing, we mean that every document returned to
    the user needs to contain all of the query terms." (L13 slide 10)
  - We use the term-at-a-time approach (L13 slide 7): build the result
    set by intersecting the posting-list document sets for each query
    term. The query term whose posting list is shortest determines the
    upper bound on matches, so a missing term short-circuits to "no
    results" immediately (L13's note that conjunctive processing is
    fastest when one term is rare).

Queries are tokenised with the same function used at index time so
case-folding is consistent; the brief states the search must be
case-insensitive ("'Good' is the same word as 'good'").

This module was developed with AI assistance (Claude); all design
choices were reviewed and adjusted by the author.
"""

from indexer import tokenise


def print_postings(index, doc_map, word):
    """Print the inverted-list entry for `word` to standard output."""
    word = (word or "").strip().lower()
    if not word:
        print("Usage: print <word>")
        return
    postings = index.get(word)
    if not postings:
        print(f"No entry for '{word}'.")
        return
    print(f"Inverted list for '{word}' ({len(postings)} document(s)):")
    for doc_id, stats in postings.items():
        url = doc_map.get(doc_id, f"<unknown doc {doc_id}>")
        frequency = stats["frequency"]
        positions = stats["positions"]
        print(f"  {url}")
        print(f"    frequency = {frequency}")
        print(f"    positions = {positions}")


def find(index, doc_map, query_terms):
    """Return URLs of pages containing every term in `query_terms`.

    `query_terms` is a list of raw strings (e.g. command-line arguments).
    Each is tokenised with the indexer's tokeniser so multi-character
    arguments such as "good!" are handled the same way the index
    handled them at build time.
    """
    if not query_terms:
        return []

    tokens = []
    for raw in query_terms:
        tokens.extend(tokenise(raw))
    if not tokens:
        return []

    # Conjunctive processing: intersect the doc-id sets of each term's
    # posting list. If any term has no posting list, the conjunction is
    # empty.
    doc_id_sets = []
    for token in tokens:
        postings = index.get(token)
        if not postings:
            return []
        doc_id_sets.append(set(postings.keys()))

    matched_ids = set.intersection(*doc_id_sets)
    return [doc_map[doc_id] for doc_id in matched_ids if doc_id in doc_map]
