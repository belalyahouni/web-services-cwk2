"""
Query processing for the COMP3011 search engine tool.

`print_postings` shows the raw posting list for a term (the `print`
command in the brief).

`find` implements multi-word search with conjunctive processing as
described in Lecture 13 (Query Processing) and ranks the surviving
documents by TF-IDF score (lnc.ltc-style):

  - Conjunctive intersection: a document is a candidate only if it
    contains every query term (L13 slide 10 "By conjunctive processing,
    we mean that every document returned to the user needs to contain
    all of the query terms").
  - Term-at-a-time evaluation: intersect the doc-id sets of each term's
    posting list (L13 slide 7). A missing term short-circuits to no
    results.
  - TF-IDF scoring (Manning et al., "Introduction to Information
    Retrieval", chapter 6; consistent with L12/L13 weighting):
        tf(t,d)    = 1 + log10(freq(t,d))            (sublinear tf)
        idf(t)     = log10(N / df(t))                (inverse doc freq)
        score(q,d) = sum over t in q of tf(t,d) * idf(t)
    where N is the number of indexed documents and df(t) is the number
    of documents containing t. Sublinear tf prevents very-frequent
    terms from dominating; idf damps common terms.
  - Results are sorted by score descending; URL is the deterministic
    tie-breaker so identical-scoring docs come back in a stable order
    (important for reproducible CLI output and tests).

Queries are tokenised with the same function used at index time so
case-folding is consistent; the brief states the search must be
case-insensitive ("'Good' is the same word as 'good'").

This module was developed with AI assistance (Claude); all design
choices were reviewed and adjusted by the author.
"""

import math

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
    """Return ranked (url, score) pairs for pages containing every query term.

    Conjunctive (AND) intersection picks the candidate documents;
    TF-IDF (lnc.ltc) ranks them. The list is sorted by score descending
    with URL as a deterministic tie-breaker. An empty query, a query
    that tokenises to nothing, or a query containing any unknown term
    returns the empty list.

    `query_terms` is a list of raw strings (e.g. command-line arguments).
    Each is tokenised the same way the index tokenised pages at build
    time so 'good!' and 'good' match identically.
    """
    if not query_terms:
        return []

    tokens = []
    for raw in query_terms:
        tokens.extend(tokenise(raw))
    if not tokens:
        return []

    # Conjunctive processing: intersect doc-id sets. A term whose
    # posting list is missing zeros the conjunction immediately (L13's
    # rare-term short-circuit).
    doc_id_sets = []
    for token in tokens:
        postings = index.get(token)
        if not postings:
            return []
        doc_id_sets.append(set(postings.keys()))

    matched_ids = set.intersection(*doc_id_sets)
    if not matched_ids:
        return []

    return _rank_by_tfidf(index, doc_map, tokens, matched_ids)


def _rank_by_tfidf(index, doc_map, tokens, matched_ids):
    """Score each matched document by summed TF-IDF and sort.

    Sublinear tf and standard idf; see module docstring for the formula.
    Tokens may repeat (e.g. find good good): each occurrence contributes
    independently, which is consistent with treating the query as a bag
    of terms.
    """
    num_docs = len(doc_map)
    # Pre-compute idf per unique token; df is the same for every doc.
    idf_cache = {}
    for token in set(tokens):
        df = len(index[token])
        idf_cache[token] = math.log10(num_docs / df) if df > 0 else 0.0

    scored = []
    for doc_id in matched_ids:
        url = doc_map.get(doc_id)
        if url is None:
            continue
        score = 0.0
        for token in tokens:
            entry = index[token][doc_id]
            tf = 1.0 + math.log10(entry["frequency"])
            score += tf * idf_cache[token]
        scored.append((url, score))

    scored.sort(key=lambda pair: (-pair[1], pair[0]))
    return scored
