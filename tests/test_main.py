"""
Tests for the CLI shell in src/main.py.

Covers each command function in isolation (cmd_build, cmd_load,
cmd_print, cmd_find), the REPL loop's command dispatch, error paths
(no index loaded, missing file, unknown command, malformed input), and
graceful exit. main.Crawler and main.INDEX_PATH are monkeypatched so
the tests run offline and never touch the real data/ directory.
"""

import builtins
import io
import os
from contextlib import redirect_stdout

import pytest

import main


# ---- helpers ------------------------------------------------------------

class _FakeCrawler:
    """Stands in for crawler.Crawler with a fixed page set."""

    def __init__(self, *_args, **_kwargs):
        self.pages = {
            "http://x/p1": "<html>good morning friends</html>",
            "http://x/p2": "<html>good afternoon</html>",
        }

    def crawl(self, *_args, **_kwargs):
        return self.pages


class _EmptyCrawler:
    """A crawler that returns nothing - simulates the unhappy path."""

    def __init__(self, *_args, **_kwargs):
        pass

    def crawl(self, *_args, **_kwargs):
        return {}


def _capture(callable_, *args, **kwargs):
    out = io.StringIO()
    with redirect_stdout(out):
        callable_(*args, **kwargs)
    return out.getvalue()


# ---- cmd_build ----------------------------------------------------------

def test_cmd_build_populates_state_and_writes_file(monkeypatch, tmp_path):
    index_path = tmp_path / "index.json"
    monkeypatch.setattr(main, "Crawler", _FakeCrawler)
    monkeypatch.setattr(main, "INDEX_PATH", str(index_path))

    state = {}
    output = _capture(main.cmd_build, state)

    assert "index" in state
    assert "doc_map" in state
    assert index_path.exists()
    assert "Crawled 2 pages" in output
    assert "Saved to" in output


def test_cmd_build_aborts_when_crawl_returns_no_pages(monkeypatch, tmp_path):
    index_path = tmp_path / "index.json"
    monkeypatch.setattr(main, "Crawler", _EmptyCrawler)
    monkeypatch.setattr(main, "INDEX_PATH", str(index_path))

    state = {}
    output = _capture(main.cmd_build, state)

    assert "No pages were crawled" in output
    assert "index" not in state
    assert not index_path.exists()


# ---- cmd_load -----------------------------------------------------------

def test_cmd_load_reports_missing_file(monkeypatch, tmp_path):
    monkeypatch.setattr(main, "INDEX_PATH", str(tmp_path / "missing.json"))

    state = {}
    output = _capture(main.cmd_load, state)

    assert "No saved index found" in output
    assert "index" not in state


def test_cmd_load_populates_state_from_existing_file(monkeypatch, tmp_path):
    """A round-trip: build with cmd_build then re-load with cmd_load."""
    index_path = tmp_path / "index.json"
    monkeypatch.setattr(main, "Crawler", _FakeCrawler)
    monkeypatch.setattr(main, "INDEX_PATH", str(index_path))

    # Populate.
    _capture(main.cmd_build, {})

    # Now load into a fresh state dict.
    state = {}
    output = _capture(main.cmd_load, state)

    assert "index" in state
    assert "doc_map" in state
    assert "Loaded index" in output


# ---- cmd_print ----------------------------------------------------------

def test_cmd_print_warns_when_no_index_is_loaded():
    output = _capture(main.cmd_print, {}, ["good"])
    assert "No index loaded" in output


def test_cmd_print_with_no_args_prints_usage():
    output = _capture(main.cmd_print, {"index": {}, "doc_map": {}}, [])
    assert "Usage" in output


def test_cmd_print_with_extra_args_warns_and_uses_first(monkeypatch):
    """Extra args after the first word are ignored with a note."""
    state = {
        "index": {"good": {"0": {"frequency": 1, "positions": [0]}}},
        "doc_map": {"0": "http://x/p"},
    }
    output = _capture(main.cmd_print, state, ["good", "extra"])
    assert "ignoring extra" in output.lower()
    assert "good" in output


def test_cmd_print_outputs_posting_data():
    state = {
        "index": {"good": {"0": {"frequency": 2, "positions": [3, 9]}}},
        "doc_map": {"0": "http://x/p"},
    }
    output = _capture(main.cmd_print, state, ["good"])
    assert "frequency = 2" in output
    assert "positions = [3, 9]" in output
    assert "http://x/p" in output


# ---- cmd_find -----------------------------------------------------------

def test_cmd_find_warns_when_no_index_is_loaded():
    output = _capture(main.cmd_find, {}, ["good"])
    assert "No index loaded" in output


def test_cmd_find_with_no_args_prints_usage():
    output = _capture(main.cmd_find, {"index": {}, "doc_map": {}}, [])
    assert "Usage" in output


def test_cmd_find_reports_no_matches_when_query_misses():
    state = {
        "index": {"good": {"0": {"frequency": 1, "positions": [0]}}},
        "doc_map": {"0": "http://x/p"},
    }
    output = _capture(main.cmd_find, state, ["nonsense"])
    assert "No matching pages" in output


def test_cmd_find_lists_matching_urls():
    state = {
        "index": {
            "good": {
                "0": {"frequency": 1, "positions": [0]},
                "1": {"frequency": 1, "positions": [0]},
            }
        },
        "doc_map": {"0": "http://x/p1", "1": "http://x/p2"},
    }
    output = _capture(main.cmd_find, state, ["good"])
    assert "http://x/p1" in output
    assert "http://x/p2" in output
    assert "matching page(s)" in output


# ---- repl loop ----------------------------------------------------------

def _scripted_input(lines):
    """Return an input() replacement that yields each line in turn,
    raising EOFError after the last line so the REPL exits cleanly."""
    iterator = iter(lines)

    def fake_input(_prompt=""):
        try:
            return next(iterator)
        except StopIteration:
            raise EOFError

    return fake_input


def test_repl_quit_command_exits(monkeypatch):
    monkeypatch.setattr(builtins, "input", _scripted_input(["quit"]))
    output = _capture(main.repl)
    assert "COMP3011 Search Engine Tool" in output


def test_repl_help_command_prints_help(monkeypatch):
    monkeypatch.setattr(builtins, "input", _scripted_input(["help", "quit"]))
    output = _capture(main.repl)
    assert "build" in output
    assert "load" in output
    assert "print" in output
    assert "find" in output


def test_repl_unknown_command_is_reported(monkeypatch):
    monkeypatch.setattr(builtins, "input", _scripted_input(["frobnicate", "quit"]))
    output = _capture(main.repl)
    assert "Unknown command" in output


def test_repl_blank_lines_are_ignored(monkeypatch):
    monkeypatch.setattr(builtins, "input", _scripted_input(["", "   ", "quit"]))
    # Should not raise - just keep looping past the blank lines.
    _capture(main.repl)


def test_repl_eof_exits_cleanly(monkeypatch):
    """Hitting EOF (Ctrl-D) at the prompt should exit without crashing."""
    monkeypatch.setattr(builtins, "input", _scripted_input([]))  # immediate EOF
    _capture(main.repl)


def test_repl_malformed_quoting_is_reported(monkeypatch):
    """An unterminated quoted string should print a parse error, not crash."""
    monkeypatch.setattr(
        builtins, "input", _scripted_input(["find \"unterminated", "quit"])
    )
    output = _capture(main.repl)
    assert "Could not parse command" in output


def test_repl_dispatches_print_command(monkeypatch, tmp_path):
    """End-to-end: build then print 'good' through the REPL."""
    index_path = tmp_path / "index.json"
    monkeypatch.setattr(main, "Crawler", _FakeCrawler)
    monkeypatch.setattr(main, "INDEX_PATH", str(index_path))
    monkeypatch.setattr(
        builtins, "input", _scripted_input(["build", "print good", "quit"])
    )
    output = _capture(main.repl)
    assert "Crawled 2 pages" in output
    assert "Inverted list for 'good'" in output


def test_repl_dispatches_load_command(monkeypatch, tmp_path):
    """End-to-end through the REPL: build (so a file exists), then
    explicitly run `load` to read it back into a fresh state."""
    index_path = tmp_path / "index.json"
    monkeypatch.setattr(main, "Crawler", _FakeCrawler)
    monkeypatch.setattr(main, "INDEX_PATH", str(index_path))
    monkeypatch.setattr(
        builtins,
        "input",
        _scripted_input(["build", "load", "find good", "quit"]),
    )
    output = _capture(main.repl)
    assert "Loaded index" in output
    assert "matching page(s)" in output


def test_repl_dispatches_find_command(monkeypatch, tmp_path):
    """End-to-end: build then find through the REPL."""
    index_path = tmp_path / "index.json"
    monkeypatch.setattr(main, "Crawler", _FakeCrawler)
    monkeypatch.setattr(main, "INDEX_PATH", str(index_path))
    monkeypatch.setattr(
        builtins,
        "input",
        _scripted_input(["build", "find good friends", "quit"]),
    )
    output = _capture(main.repl)
    assert "matching page(s)" in output
    assert "http://x/p1" in output
