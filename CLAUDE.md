# CLAUDE.md

Project context for future Claude sessions on this repository.

## What this project is

COMP3011 (Web Services and Web Data) Coursework 2 — a small command-line
**search engine** for `https://quotes.toscrape.com/`. Crawl the site, build
an inverted index over every word occurrence, and answer single- and
multi-word queries against the loaded index. Implemented in Python 3
against the COMP3011 lecture material (L9 Web Crawling, L11 Parsing &
Tokenisation, L12 Indexing, L13 Query Processing).

GenAI declaration: code was developed with Claude as an assistive pair —
declared in every module docstring and in the README.

## Repository layout

Mandated by the brief:

```
code/
  src/   { crawler.py, indexer.py, search.py, main.py }
  tests/ { test_crawler.py, test_indexer.py, test_search.py,
           test_main.py, test_integration.py }
  data/  { index.json — produced by `build` }
  requirements.txt
  README.md
  CLAUDE.md
  conftest.py        # adds src/ to sys.path for pytest
  .gitignore
```

`src/` is **not** a package (no `__init__.py`). Internal imports are flat
(`from crawler import Crawler`). `main.py` prepends its own directory to
`sys.path` at startup so both `python src/main.py` and `python -m src.main`
work.

## Modules and what they do

### `src/crawler.py` — `Crawler`
- BFS from a seed URL using a frontier queue and a visited set.
- Restricted to the seed host (no off-site crawling).
- **6-second politeness window** between successive requests (mandated
  by the brief, L9 slide 12).
- Strips URL fragments so `/a` and `/a#section` are one page.
- Skips non-200 responses, non-HTML content types, network errors, and
  non-http schemes (`mailto:`, `javascript:`, `tel:`).
- `crawl(max_pages=None, verbose=True)` returns `{url: html}`. The HTML
  is consumed by the indexer and not retained afterwards.

### `src/indexer.py`
- `extract_text(html)` — BeautifulSoup parses the HTML, strips
  `<script>`, `<style>`, `<noscript>`, and returns visible text (L11
  pass 1).
- `tokenise(text)` — lowercase the text and `re.findall` runs of
  `[a-z0-9]+` (L11 pass 2). **All words are kept** — no stopword
  removal, no stemming. This is a deliberate choice (project decision
  given to Claude) so that pure-stopword queries like *"to be or not
  to be"* still match (L11 caveat).
- `build_index(pages)` returns `(index, doc_map)`:
  - `index`: `{term: {doc_id: {"frequency": int, "positions": [int]}}}`
  - `doc_map`: `{doc_id: url}` where `doc_id` is a stringified ordinal
    (so the structure round-trips through JSON).
  - Positions are 0-indexed token positions on the page (L12 slide 15).
- `save_index(path, index, doc_map)` / `load_index(path)` — single JSON
  file holding both halves under `"index"` and `"doc_map"` keys (L12
  slide 25 single-inverted-file approach).

### `src/search.py`
- `print_postings(index, doc_map, word)` — dumps the posting list for
  one term (URLs + frequencies + positions). Case-insensitive. Reports
  "No entry" for missing terms.
- `find(index, doc_map, query_terms)` — multi-word **conjunctive**
  search per L13. Tokenises the query the same way the indexer
  tokenised documents, intersects the doc-id sets of each term's
  posting list, returns matching URLs. Any missing term short-circuits
  to `[]`. Order-independent. Phrase queries are *not* implemented —
  positions are stored but not consulted.

### `src/main.py`
- Interactive REPL exposing `build`, `load`, `print`, `find`, plus
  `help` and `quit`/`exit` (and EOF). Uses `shlex.split` so quoted
  multi-word arguments work.
- Module constants: `SEED_URL = "https://quotes.toscrape.com/"`,
  `INDEX_PATH` resolves to `data/index.json` at the repo root.
- Each command function (`cmd_build`, `cmd_load`, `cmd_print`,
  `cmd_find`) takes a `state` dict that holds the in-memory `index`
  and `doc_map`. Re-running `build` or `load` replaces them.
- Reports common error states (no index loaded, missing file, unknown
  command, malformed input) without crashing.

## Conventions and decisions baked in

| Decision | Where it lives |
| --- | --- |
| All words kept (no stopwords, no stemming) | `indexer.tokenise` |
| Lowercase tokens, alphanumeric only | `indexer.TOKEN_PATTERN` |
| 6-second politeness | `crawler.DEFAULT_POLITENESS_SECONDS` |
| Single-host crawl | `crawler.Crawler._extract_links` (host filter) |
| Single JSON file index | `indexer.save_index` |
| Conjunctive (AND) multi-word search | `search.find` |
| String doc-ids for JSON safety | `indexer.build_index` |
| Doc-ids assigned in crawl-discovery order | `indexer.build_index` |

## Running

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python src/main.py            # or: python -m src.main
```

Then in the shell: `build`, `load`, `print <word>`, `find <terms...>`.

## Tests

```bash
pytest                          # 66 tests, ~1s
pytest -v                       # show each test name
pytest tests/test_main.py       # single file
```

Coverage: 99% line coverage across `src/` (the 2 missed lines are a
defensive duplicate-visit guard in the crawler and the
`if __name__ == "__main__"` block in `main.py`).

The suite mocks the network — no live HTTP. `conftest.py` at the repo
root puts `src/` on `sys.path` so tests can do `from crawler import ...`
directly. Tests must be run via `pytest`, not `python tests/test_*.py`.

Categories present (per the 70-79 grade band's "unit, integration, and
performance tests" requirement):

- **Unit:** `test_crawler.py`, `test_indexer.py`, `test_search.py`,
  `test_main.py`.
- **Integration:** `test_integration.py` drives the full pipeline
  through every module boundary with HTTP mocked.
- **Performance:** `test_integration.py` also asserts that building an
  index from 200 synthetic pages, running 100 finds against it, and
  tokenising a 1MB string each complete in under a second.

## Git workflow used

Linear history, one logical unit per commit. Commit messages reference
the relevant lecture (L9, L11, L12, L13) where applicable. Co-authored
with Claude per the GenAI declaration policy.
