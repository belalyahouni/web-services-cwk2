"""
Command-line shell for the COMP3011 search engine tool.

Provides the four commands required by the brief:

    > build           crawl the target site, build the inverted index,
                      and save it to data/index.json
    > load            load a previously-built index from data/index.json
    > print <word>    print the inverted-list entry for a single word
    > find <terms...> return all pages containing every query term

`quit` (or Ctrl-D) exits the shell.

This module was developed with AI assistance (Claude); all design
choices were reviewed and adjusted by the author.
"""

import os
import shlex
import sys

# Make the imports below resolve regardless of how this file is launched.
# `python src/main.py` already puts src/ first on sys.path; `python -m
# src.main` does not (it puts the cwd first), so we add src/ explicitly.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from crawler import Crawler
from indexer import build_index, load_index, save_index
from search import find, print_postings


SEED_URL = "https://quotes.toscrape.com/"

# data/index.json sits at the repo root, alongside src/, tests/, etc.,
# matching the directory layout mandated by the brief.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INDEX_PATH = os.path.join(REPO_ROOT, "data", "index.json")


def cmd_build(state):
    """Crawl the target site, build the inverted index, persist to disk."""
    print(f"Crawling {SEED_URL} (politeness window: 6 seconds)")
    crawler = Crawler(SEED_URL)
    pages = crawler.crawl()
    if not pages:
        print("No pages were crawled. Aborting.")
        return
    print(f"Crawled {len(pages)} pages. Building inverted index...")
    index, doc_map = build_index(pages)
    os.makedirs(os.path.dirname(INDEX_PATH), exist_ok=True)
    save_index(INDEX_PATH, index, doc_map)
    state["index"] = index
    state["doc_map"] = doc_map
    print(f"Index built ({len(index)} unique terms across {len(doc_map)} documents).")
    print(f"Saved to {INDEX_PATH}")


def cmd_load(state):
    """Load the previously saved index from disk."""
    if not os.path.exists(INDEX_PATH):
        print(f"No saved index found at {INDEX_PATH}. Run 'build' first.")
        return
    index, doc_map = load_index(INDEX_PATH)
    state["index"] = index
    state["doc_map"] = doc_map
    print(f"Loaded index ({len(index)} unique terms across {len(doc_map)} documents).")


def cmd_print(state, args):
    """Print the inverted-list entry for a single word."""
    if not args:
        print("Usage: print <word>")
        return
    if "index" not in state:
        print("No index loaded. Run 'load' or 'build' first.")
        return
    if len(args) > 1:
        print("Note: 'print' takes one word; ignoring extra arguments.")
    print_postings(state["index"], state["doc_map"], args[0])


def cmd_find(state, args):
    """Find pages containing all given query terms, ranked by TF-IDF."""
    if not args:
        print("Usage: find <term> [<term> ...]")
        return
    if "index" not in state:
        print("No index loaded. Run 'load' or 'build' first.")
        return
    matches = find(state["index"], state["doc_map"], args)
    if not matches:
        print("No matching pages.")
        return
    print(f"{len(matches)} matching page(s) (ranked by TF-IDF):")
    for url, score in matches:
        print(f"  [{score:.4f}] {url}")


HELP_TEXT = (
    "Commands:\n"
    "  build              Crawl the target site and build the index.\n"
    "  load               Load a previously-built index from disk.\n"
    "  print <word>       Show the inverted-list entry for a word.\n"
    "  find <term> ...    Find pages containing all given terms.\n"
    "  help               Show this help message.\n"
    "  quit               Exit the shell."
)


def repl():
    """Run the interactive command shell."""
    state = {}
    print("COMP3011 Search Engine Tool")
    print("Type 'help' for commands, 'quit' to exit.")
    while True:
        try:
            line = input("> ")
        except (EOFError, KeyboardInterrupt):
            print()
            break

        line = line.strip()
        if not line:
            continue

        try:
            tokens = shlex.split(line)
        except ValueError as err:
            print(f"Could not parse command: {err}")
            continue

        command = tokens[0].lower()
        args = tokens[1:]

        if command in ("quit", "exit"):
            break
        elif command == "help":
            print(HELP_TEXT)
        elif command == "build":
            cmd_build(state)
        elif command == "load":
            cmd_load(state)
        elif command == "print":
            cmd_print(state, args)
        elif command == "find":
            cmd_find(state, args)
        else:
            print(f"Unknown command: {command!r}. Type 'help' for a list.")


if __name__ == "__main__":
    repl()
