"""
Web crawler for the COMP3011 search engine tool.

Implements a breadth-first traversal of a single website following the
crawling process described in Lecture 9 (Web Crawling):

  - Maintain a frontier (queue) of URLs to visit.
  - Pop a URL, fetch the page, parse it for outgoing links, add new ones
    to the frontier (the visited set ensures we never revisit a page).
  - Enforce a politeness window of at least 6 seconds between successive
    requests to the same web server (mandated by the coursework brief
    and motivated in L9 slide 12 "Politeness Policies").
  - Stay within the seed URL's host (so we crawl the website, not the
    whole web).

This module was developed with AI assistance (Claude); all design
choices were reviewed and adjusted by the author.
"""

from __future__ import annotations

import time
from collections import deque
from urllib.parse import urldefrag, urljoin, urlparse

import requests
from bs4 import BeautifulSoup


DEFAULT_POLITENESS_SECONDS = 6.0
DEFAULT_TIMEOUT_SECONDS = 10.0
DEFAULT_USER_AGENT = "COMP3011-SearchTool/1.0 (educational crawler)"


class Crawler:
    """Breadth-first crawler restricted to a single host."""

    def __init__(
        self,
        seed_url: str,
        politeness: float = DEFAULT_POLITENESS_SECONDS,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        user_agent: str = DEFAULT_USER_AGENT,
    ) -> None:
        self.seed_url = seed_url
        self.politeness = politeness
        self.timeout = timeout
        self.host = urlparse(seed_url).netloc
        self._last_fetch_time = 0.0
        self.session = requests.Session()
        self.session.headers["User-Agent"] = user_agent

    def crawl(self, max_pages: int | None = None, verbose: bool = True) -> dict[str, str]:
        """Crawl the website starting from the seed URL.

        Returns a dict mapping each successfully fetched URL to its raw
        HTML body. Pages that returned an error or were not HTML are
        silently skipped (handled gracefully per L9 challenges and the
        coursework's "handle errors gracefully" tip).
        """
        # deque gives O(1) popleft for the BFS dequeue; a plain list would
        # be O(n) per pop because every remaining element shifts left.
        frontier: deque[str] = deque([self._normalise(self.seed_url)])
        queued: set[str] = set(frontier)
        visited: set[str] = set()
        pages: dict[str, str] = {}

        while frontier:
            url = frontier.popleft()
            if url in visited:
                continue
            visited.add(url)

            if verbose:
                print(f"  fetching {url}")
            html = self._fetch(url)
            if html is None:
                continue
            pages[url] = html

            if max_pages is not None and len(pages) >= max_pages:
                break

            for link in self._extract_links(url, html):
                if link in visited or link in queued:
                    continue
                queued.add(link)
                frontier.append(link)

        return pages

    def _fetch(self, url: str) -> str | None:
        """Fetch a single URL, respecting the politeness window.

        Returns the response body as a string on success, or None on any
        failure (non-200 status, network error, non-HTML content type).
        The politeness clock is updated whether or not the request
        succeeded, so a flurry of failed requests cannot bypass it.
        """
        self._wait_for_politeness()
        try:
            response = self.session.get(url, timeout=self.timeout)
        except requests.RequestException:
            self._last_fetch_time = time.monotonic()
            return None

        self._last_fetch_time = time.monotonic()
        if response.status_code != 200:
            return None
        content_type = response.headers.get("Content-Type", "").lower()
        if "html" not in content_type and content_type != "":
            return None
        return response.text

    def _wait_for_politeness(self) -> None:
        """Sleep until at least `self.politeness` seconds have passed
        since the previous request (L9 politeness policy)."""
        elapsed = time.monotonic() - self._last_fetch_time
        if elapsed < self.politeness:
            time.sleep(self.politeness - elapsed)

    def _extract_links(self, base_url: str, html: str) -> list[str]:
        """Parse anchor tags and return absolute, host-internal URLs.

        Fragments are stripped (so /a and /a#section count as the same
        page) and only http(s) URLs on the seed host are kept.
        """
        soup = BeautifulSoup(html, "html.parser")
        links: list[str] = []
        for anchor in soup.find_all("a", href=True):
            absolute = urljoin(base_url, anchor["href"])
            absolute = self._normalise(absolute)
            parsed = urlparse(absolute)
            if parsed.scheme not in ("http", "https"):
                continue
            if parsed.netloc != self.host:
                continue
            links.append(absolute)
        return links

    @staticmethod
    def _normalise(url: str) -> str:
        """Drop the URL fragment so duplicates are detected reliably."""
        defragged, _ = urldefrag(url)
        return defragged
