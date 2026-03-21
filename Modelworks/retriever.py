"""
Treuno 125M â€”  Web Retriever
Real-time web search to ground every inference call with live documentation.

Primary backend: DuckDuckGo (no API key required)
Optional backend: Serper.dev (higher quality, requires SERPER_API_KEY env var)

Usage:
    retriever = ModelworksRetriever(max_results=5)
    results = retriever.search("python requests library POST example")
    for r in results:
        print(r.title, r.url, r.snippet[:200])
"""

from __future__ import annotations
import os
import time
import logging
from dataclasses import dataclass, field
from typing import List, Optional

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str
    source: str = "duckduckgo"    # or "serper"
    score: float = 1.0            # relevance score (0â€“1)


class ModelworksRetriever:
    """
    Real-time web retriever for the Modelworks system.

    At every Treuno inference call:
      1. Parse the query for library / API names
      2. Search the web for current documentation
      3. Return top-K snippets for RAG injection

    This eliminates hallucinated APIs by grounding the model
    in live documentation rather than stale training data.
    """

    def __init__(
        self,
        max_results: int = 5,
        timeout: int = 8,
        serper_api_key: Optional[str] = None,
    ):
        self.max_results = max_results
        self.timeout = timeout
        self.serper_api_key = serper_api_key or os.environ.get("SERPER_API_KEY")
        self._last_search_time: float = 0.0
        self._rate_limit_seconds: float = 1.0  # DuckDuckGo rate limit guard

    def search(self, query: str, num_results: Optional[int] = None) -> List[SearchResult]:
        """
        Search the web and return ranked results.

        Args:
            query:       Natural language or code-specific search query
            num_results: Override default max_results

        Returns:
            List of SearchResult sorted by relevance
        """
        k = num_results or self.max_results
        # Prefer Serper if key available (higher quality, higher rate limits)
        if self.serper_api_key:
            try:
                return self._serper_search(query, k)
            except Exception as e:
                logger.warning(f"Serper search failed: {e}. Falling back to DuckDuckGo.")
        return self._ddg_search(query, k)

    def search_for_code_query(self, query: str) -> List[SearchResult]:
        """
        Code-optimized search: appends "documentation example" to query
        and filters results to prioritize official docs and GitHub.
        """
        enhanced_query = f"{query} documentation example site:docs.python.org OR site:github.com OR site:stackoverflow.com"
        results = self.search(enhanced_query)
        # Fallback without site filter if no results
        if not results:
            results = self.search(f"{query} documentation example")
        return results

    # â”€â”€ DuckDuckGo backend â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def _ddg_search(self, query: str, k: int) -> List[SearchResult]:
        """Search using duckduckgo-search library (no API key needed)."""
        # Rate limit
        elapsed = time.time() - self._last_search_time
        if elapsed < self._rate_limit_seconds:
            time.sleep(self._rate_limit_seconds - elapsed)

        try:
            from duckduckgo_search import DDGS
            results = []
            with DDGS(timeout=self.timeout) as ddgs:
                for r in ddgs.text(query, max_results=k):
                    results.append(SearchResult(
                        title=r.get("title", ""),
                        url=r.get("href", ""),
                        snippet=r.get("body", ""),
                        source="duckduckgo",
                    ))
            self._last_search_time = time.time()
            return results
        except ImportError:
            logger.error("duckduckgo-search not installed. Run: pip install duckduckgo-search")
            return []
        except Exception as e:
            logger.error(f"DuckDuckGo search failed: {e}")
            return []

    # â”€â”€ Serper backend â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def _serper_search(self, query: str, k: int) -> List[SearchResult]:
        """Search using Serper.dev API (requires SERPER_API_KEY)."""
        import requests
        payload = {"q": query, "num": k}
        headers = {
            "X-API-KEY": self.serper_api_key,
            "Content-Type": "application/json",
        }
        resp = requests.post(
            "https://google.serper.dev/search",
            json=payload,
            headers=headers,
            timeout=self.timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        results = []
        for item in data.get("organic", []):
            results.append(SearchResult(
                title=item.get("title", ""),
                url=item.get("link", ""),
                snippet=item.get("snippet", ""),
                source="serper",
            ))
        return results

    # â”€â”€ Snippet fetching â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def fetch_page_text(self, url: str, max_chars: int = 4000) -> str:
        """
        Fetch and extract plain text from a URL.
        Used to get full content beyond the search snippet.
        """
        try:
            import requests
            from bs4 import BeautifulSoup
            resp = requests.get(url, timeout=self.timeout, headers={
                "User-Agent": "Mozilla/5.0 (Treuno/0.1 ModelworksBot)"
            })
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "lxml")
            # Remove script/style tags
            for tag in soup(["script", "style", "nav", "footer", "header"]):
                tag.decompose()
            text = soup.get_text(separator="\n", strip=True)
            # Trim to max_chars
            return text[:max_chars]
        except Exception as e:
            logger.warning(f"Failed to fetch {url}: {e}")
            return ""

    def __repr__(self) -> str:
        backend = "serper" if self.serper_api_key else "duckduckgo"
        return f"ModelworksRetriever(backend={backend}, max_results={self.max_results})"
