from __future__ import annotations

import html
import re
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Dict, List, Sequence


TOOL_CALL_RE = re.compile(r"^\s*TOOL:web_search\s*:\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE)
AUTO_WEB_RE = re.compile(
    r"\b(latest|recent|today|current|news|look up|lookup|search|web|docs|documentation|version|release|price|who is|what is new)\b",
    re.IGNORECASE,
)
LITE_RESULT_RE = re.compile(
    r"<a rel=\"nofollow\" href=\"(?P<href>[^\"]+)\" class='result-link'>(?P<title>.*?)</a>.*?"
    r"(?:<td class='result-snippet'>(?P<snippet>.*?)</td>.*?)?"
    r"<span class='link-text'>(?P<domain>.*?)</span>",
    re.IGNORECASE | re.DOTALL,
)
HTML_RESULT_RE = re.compile(
    r'class="result__a"\s+href="(?P<href>[^"]+)">(?P<title>.*?)</a>.*?'
    r'class="result__snippet"[^>]*>(?P<snippet>.*?)</',
    re.IGNORECASE | re.DOTALL,
)


@dataclass
class ToolEvent:
    name: str
    query: str
    results: List[Dict[str, str]]
    source: str = "web_search"

    def to_dict(self) -> Dict[str, object]:
        return {
            "name": self.name,
            "query": self.query,
            "source": self.source,
            "results": self.results,
        }


def parse_tool_calls(text: str) -> List[str]:
    return [match.group(1).strip() for match in TOOL_CALL_RE.finditer(str(text or "")) if match.group(1).strip()]


def strip_tool_calls(text: str) -> str:
    lines = [line for line in str(text or "").splitlines() if not TOOL_CALL_RE.match(line)]
    return "\n".join(lines).strip()


def should_offer_web_search(prompt: str) -> bool:
    return bool(AUTO_WEB_RE.search(str(prompt or "")))


def _clean_html_text(value: str) -> str:
    cooked = re.sub(r"<[^>]+>", " ", str(value or ""))
    cooked = html.unescape(cooked)
    cooked = re.sub(r"\s+", " ", cooked).strip()
    return cooked


def _unwrap_ddg_href(href: str) -> str:
    cooked = html.unescape(str(href or "").strip())
    if cooked.startswith("//"):
        cooked = "https:" + cooked
    parsed = urllib.parse.urlparse(cooked)
    query = urllib.parse.parse_qs(parsed.query)
    if "uddg" in query and query["uddg"]:
        return query["uddg"][0]
    return cooked


class WebSearchTool:
    def __init__(self, user_agent: str = "SupermixStudio/1.0") -> None:
        self.user_agent = user_agent

    def _fetch_text(self, url: str) -> str:
        request = urllib.request.Request(url, headers={"User-Agent": self.user_agent})
        with urllib.request.urlopen(request, timeout=25) as response:
            return response.read().decode("utf-8", errors="ignore")

    def _parse_lite(self, text: str, max_results: int) -> List[Dict[str, str]]:
        results: List[Dict[str, str]] = []
        for match in LITE_RESULT_RE.finditer(text):
            title = _clean_html_text(match.group("title"))
            href = _unwrap_ddg_href(match.group("href"))
            snippet = _clean_html_text(match.group("snippet") or "")
            domain = _clean_html_text(match.group("domain") or "")
            if not title or not href:
                continue
            results.append({"title": title, "url": href, "snippet": snippet, "domain": domain})
            if len(results) >= max_results:
                break
        return results

    def _parse_html(self, text: str, max_results: int) -> List[Dict[str, str]]:
        results: List[Dict[str, str]] = []
        for match in HTML_RESULT_RE.finditer(text):
            title = _clean_html_text(match.group("title"))
            href = _unwrap_ddg_href(match.group("href"))
            snippet = _clean_html_text(match.group("snippet") or "")
            domain = urllib.parse.urlparse(href).netloc
            if not title or not href:
                continue
            results.append({"title": title, "url": href, "snippet": snippet, "domain": domain})
            if len(results) >= max_results:
                break
        return results

    def search(self, query: str, max_results: int = 5) -> ToolEvent:
        cooked_query = re.sub(r"\s+", " ", str(query or "").strip())[:220]
        if not cooked_query:
            raise ValueError("web_search query is empty")
        encoded = urllib.parse.quote_plus(cooked_query)
        results = self._parse_lite(self._fetch_text(f"https://lite.duckduckgo.com/lite/?q={encoded}"), max_results=max_results)
        if not results:
            results = self._parse_html(self._fetch_text(f"https://html.duckduckgo.com/html/?q={encoded}"), max_results=max_results)
        return ToolEvent(name="web_search", query=cooked_query, results=results[:max_results], source="duckduckgo")


def format_tool_results(events: Sequence[ToolEvent]) -> str:
    blocks: List[str] = []
    for event in events:
        rows = [f"Web search query: {event.query}"]
        for idx, item in enumerate(event.results[:5], start=1):
            title = item.get("title") or item.get("url") or "result"
            domain = item.get("domain") or urllib.parse.urlparse(item.get("url") or "").netloc
            snippet = item.get("snippet") or ""
            if snippet:
                rows.append(f"{idx}. {title} [{domain}] - {snippet}")
            else:
                rows.append(f"{idx}. {title} [{domain}]")
        blocks.append("\n".join(rows))
    return "\n\n".join(blocks).strip()
