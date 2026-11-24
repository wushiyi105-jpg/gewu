"""零依赖联网检索:直接抓搜索引擎的 HTML 结果页,不需要任何 API key。

为什么是「多引擎」:
  - 搜狗国内可达,摘要完整,但短时间高频请求会被限流;
  - 360 搜索国内稳定,结果里带 data-mdurl 真实直链,是搜狗限流时的第一替补;
  - DuckDuckGo 的 HTML 页在境外网络最稳,国内常被 TLS 层掐断;
  - Bing 的 HTML 页一般可达,但结构随 A/B 实验变动,中文查询还可能被降级成词典释义。

所以做成**自动降级 + 请求间隔**:搜狗 → 360 → DDG → Bing,
每个引擎之间有最小间隔,空结果会再试一次;全都失败抛 SearchError,
由编排层退回模型知识并诚实标注。

不引第三方库(requests / bs4 / duckduckgo-search),只用 urllib + 正则 / html.parser。
"""

from __future__ import annotations

import base64
import html
import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from html.parser import HTMLParser
from typing import Callable, Iterable

UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)
DDG_ENDPOINT = "https://html.duckduckgo.com/html/"
BING_ENDPOINT = "https://www.bing.com/search"
SO360_ENDPOINT = "https://www.so.com/s"

# 同一引擎两次请求的最小间隔:搜狗对突发的并发查询很敏感,间隔 1.6s 基本不会被限流
MIN_INTERVAL = 1.6
_last_call: dict[str, float] = {}


def _pace(name: str) -> None:
    now = time.monotonic()
    gap = MIN_INTERVAL - (now - _last_call.get(name, 0.0))
    if gap > 0:
        time.sleep(gap)
    _last_call[name] = time.monotonic()


class SearchError(Exception):
    """检索失败(网络、限流、结构变化)。"""


@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str = ""
    provider: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


# ------------------------------------------------------------------ 通用清洗


def _squash(text: str) -> str:
    return " ".join(html.unescape(text).split())


def _strip_tags(text: str) -> str:
    """去标签并压空白;行内标签(<em>)直接去掉,块级标签换成空格,免得词被粘在一起。"""
    text = re.sub(r"<(?:br|hr|/p|/div|/li|/h[1-6]|/tr|/td)\s*/?>", " ", text or "")
    return _squash(re.sub(r"<[^>]+>", "", text))


def _clean_url(href: str) -> str:
    """还原 DuckDuckGo 的跳转链接(//duckduckgo.com/l/?uddg=<encoded>)。"""
    if not href:
        return ""
    if href.startswith("//"):
        href = "https:" + href
    parts = urllib.parse.urlsplit(href)
    if parts.netloc.endswith("duckduckgo.com") and parts.path.startswith("/l/"):
        qs = urllib.parse.parse_qs(parts.query)
        if "uddg" in qs and qs["uddg"]:
            return urllib.parse.unquote(qs["uddg"][0])
    if parts.netloc.endswith("duckduckgo.com") and parts.path.startswith("/y.js"):
        return ""  # 广告位
    return href


def _decode_bing_url(href: str) -> str:
    """Bing 有时把结果包成 /ck/a?...&u=a1<base64url>。"""
    if not href:
        return ""
    if href.startswith("//"):
        href = "https:" + href
    parts = urllib.parse.urlsplit(href)
    if "bing.com" in parts.netloc and parts.path.startswith("/ck/a"):
        qs = urllib.parse.parse_qs(parts.query)
        u = (qs.get("u") or [""])[0]
        if u.startswith("a1"):
            raw = u[2:]
            raw += "=" * (-len(raw) % 4)
            try:
                return base64.urlsafe_b64decode(raw).decode("utf-8", "replace")
            except (ValueError, base64.binascii.Error):
                return ""
    if "bing.com" in parts.netloc and not parts.path.startswith("/ck/"):
        return ""  # Bing 站内链接不是我们要的来源
    return href


# ------------------------------------------------------------------ DuckDuckGo


class _DDGParser(HTMLParser):
    """从 DuckDuckGo HTML 结果页里抽出条目。"""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.results: list[SearchResult] = []
        self._current: dict | None = None
        self._capture: str | None = None
        self._buf: list[str] = []
        self._depth = 0

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        cls = attrs.get("class", "") or ""
        if tag == "a" and "result__a" in cls:
            self._flush()
            self._current = {"title": "", "url": _clean_url(attrs.get("href", "")), "snippet": ""}
            self._capture = "title"
            self._buf = []
            self._depth = 1
            return
        if self._capture:
            self._depth += 1
        if tag == "a" and "result__snippet" in cls and self._current and not self._current["snippet"]:
            self._capture = "snippet"
            self._buf = []
            self._depth = 1

    def handle_endtag(self, tag):
        if not self._capture:
            return
        self._depth -= 1
        if self._depth > 0:
            return
        text = _squash("".join(self._buf))
        if self._current is not None and self._capture == "title":
            self._current["title"] = text
        elif self._current is not None and self._capture == "snippet":
            self._current["snippet"] = text
        self._capture = None
        self._buf = []

    def handle_data(self, data):
        if self._capture:
            self._buf.append(data)

    def close(self):
        super().close()
        self._flush()

    def _flush(self):
        if self._current and self._current.get("title") and self._current.get("url"):
            self.results.append(SearchResult(provider="duckduckgo", **self._current))
        self._current = None
        self._capture = None
        self._buf = []


def parse_duckduckgo(page: str) -> list[SearchResult]:
    parser = _DDGParser()
    parser.feed(page)
    parser.close()
    return [r for r in parser.results if r.url]


# 兼容旧名字(测试与外部调用)
parse_html = parse_duckduckgo


def _duckduckgo(query: str, limit: int, timeout: int) -> list[SearchResult]:
    _pace("duckduckgo")
    data = urllib.parse.urlencode({"q": query, "kl": "cn-zh"}).encode("utf-8")
    req = urllib.request.Request(
        DDG_ENDPOINT,
        data=data,
        headers={
            "User-Agent": UA,
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        page = resp.read().decode("utf-8", errors="replace")
    results = parse_duckduckgo(page)
    if not results:
        raise SearchError("DuckDuckGo 无结果(限流或结构变化)")
    return results[:limit]


# ------------------------------------------------------------------ Bing


_BING_BLOCK = re.compile(r'<li class="b_algo".*?</li>', re.S)
_BING_TITLE = re.compile(r'<h2[^>]*>\s*<a[^>]*href="([^"]+)"[^>]*>(.*?)</a>', re.S)
_BING_SNIPPET = re.compile(r'<p[^>]*class="[^"]*b_lineclamp[a-zA-Z0-9_]*"[^>]*>(.*?)</p>', re.S)
_BING_ANY_P = re.compile(r"<p[^>]*>(.*?)</p>", re.S)


def parse_bing(page: str) -> list[SearchResult]:
    results: list[SearchResult] = []
    for block in _BING_BLOCK.findall(page):
        m = _BING_TITLE.search(block)
        if not m:
            continue
        url = _decode_bing_url(m.group(1))
        title = _strip_tags(m.group(2))
        if not url or not title:
            continue
        sm = _BING_SNIPPET.search(block) or _BING_ANY_P.search(block)
        snippet = _strip_tags(sm.group(1)) if sm else ""
        results.append(SearchResult(title=title, url=url, snippet=snippet, provider="bing"))
        if len(results) >= 10:
            break
    return results


def _bing(query: str, limit: int, timeout: int) -> list[SearchResult]:
    _pace("bing")
    url = BING_ENDPOINT + "?" + urllib.parse.urlencode({"q": query, "setlang": "zh-CN", "cc": "CN"})
    req = urllib.request.Request(
        url, headers={"User-Agent": UA, "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        page = resp.read().decode("utf-8", errors="replace")
    results = parse_bing(page)
    if not results:
        raise SearchError("Bing 无结果(结构变化或限流)")
    return results[:limit]


# ------------------------------------------------------------------ 搜狗


_SOGOU_BLOCK = re.compile(r'<div class="vrwrap"')
_SOGOU_TITLE = re.compile(r'<h3 class="vr-title">.*?<a[^>]*?href="([^"]+)"[^>]*>(.*?)</a>', re.S)
_SOGOU_SNIPPET = re.compile(r'<div class="(?:base-ellipsis|space-txt)[^"]*"[^>]*>(.*?)</div>', re.S)


def _clean_sogou_url(href: str) -> str:
    if not href:
        return ""
    if href.startswith("//"):
        href = "https:" + href
    if href.startswith("/link"):
        return "https://www.sogou.com" + href
    if href.startswith("/"):
        return ""
    return href


def parse_sogou(page: str) -> list[SearchResult]:
    starts = [m.start() for m in _SOGOU_BLOCK.finditer(page)]
    results: list[SearchResult] = []
    for i, start in enumerate(starts):
        end = starts[i + 1] if i + 1 < len(starts) else min(len(page), start + 8000)
        block = page[start:end]
        m = _SOGOU_TITLE.search(block)
        if not m:
            continue
        url = _clean_sogou_url(m.group(1))
        title = _strip_tags(m.group(2))
        if not url or not title:
            continue
        sm = _SOGOU_SNIPPET.search(block)
        snippet = _strip_tags(sm.group(1)) if sm else ""
        results.append(SearchResult(title=title, url=url, snippet=snippet, provider="sogou"))
        if len(results) >= 10:
            break
    return results


def _sogou(query: str, limit: int, timeout: int) -> list[SearchResult]:
    url = "https://www.sogou.com/web?" + urllib.parse.urlencode({"query": query})
    results: list[SearchResult] = []
    for attempt in range(2):
        _pace("sogou")
        req = urllib.request.Request(
            url, headers={"User-Agent": UA, "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8"}
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            page = resp.read().decode("utf-8", errors="replace")
        results = parse_sogou(page)
        if results:
            return results[:limit]
        if attempt == 0:
            time.sleep(1.2)  # 大概率是限流,歇一下再来
    raise SearchError("搜狗无结果(结构变化或限流)")


# ------------------------------------------------------------------ 360 搜索


_SO360_LIST = re.compile(r'<li class="res-list"')
_SO360_H3 = re.compile(r'<h3[^>]*class="[^"]*res-title[^"]*"[^>]*>(.*?)</h3>', re.S)
_SO360_A = re.compile(r"<a[^>]*>.*?</a>", re.S)
_SO360_MDURL = re.compile(r'data-mdurl="(https?://[^"]+)"')
_SO360_DESC = re.compile(r'class="res-desc"[^>]*>(.*?)</p>', re.S)


def parse_360(page: str) -> list[SearchResult]:
    """360 的结果块是 <li class="res-list">:块里带 data-mdurl 真实直链。"""
    page = re.sub(r"<style.*?</style>", "", page, flags=re.S)
    page = re.sub(r"<script.*?</script>", "", page, flags=re.S)
    starts = [m.start() for m in _SO360_LIST.finditer(page)]
    results: list[SearchResult] = []
    for i, start in enumerate(starts):
        end = starts[i + 1] if i + 1 < len(starts) else min(len(page), start + 8000)
        block = page[start:end]
        um = _SO360_MDURL.search(block)
        if not um:
            continue  # AI 卡片 / 广告位,没有真实外链
        hm = _SO360_H3.search(block)
        title = ""
        if hm:
            am = _SO360_A.search(hm.group(1))
            title = _strip_tags(am.group(0) if am else hm.group(1))
        if not title:
            continue
        dm = _SO360_DESC.search(block)
        snippet = _strip_tags(dm.group(1)) if dm else ""
        results.append(SearchResult(title=title, url=um.group(1), snippet=snippet, provider="360"))
        if len(results) >= 10:
            break
    return results


def _so360(query: str, limit: int, timeout: int) -> list[SearchResult]:
    url = SO360_ENDPOINT + "?" + urllib.parse.urlencode({"q": query})
    _pace("360")
    req = urllib.request.Request(
        url, headers={"User-Agent": UA, "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        page = resp.read().decode("utf-8", errors="replace")
    results = parse_360(page)
    if not results:
        raise SearchError("360 无结果(结构变化或限流)")
    return results[:limit]


# ------------------------------------------------------------------ 引擎调度


PROVIDERS: dict[str, Callable[[str, int, int], list[SearchResult]]] = {
    "sogou": _sogou,
    "360": _so360,
    "duckduckgo": _duckduckgo,
    "bing": _bing,
}
_provider_order = ["sogou", "360", "duckduckgo", "bing"]
_preferred: str | None = None
_dead: set[str] = set()  # 本轮已失败的引擎:同一次运行里不再重复撞墙


def _order() -> list[str]:
    override = (os.environ.get("GEWU_SEARCH_PROVIDER") or "").strip().lower()
    if override in PROVIDERS:
        return [override]
    if override in {"auto", ""} and _preferred in PROVIDERS:
        return [_preferred] + [p for p in _provider_order if p != _preferred]
    return list(_provider_order)


def _bigrams(text: str) -> set[str]:
    """中文二元组,用来判断「标题和查询到底沾不沾边」。"""
    chars = re.sub(r"[^\u4e00-\u9fffA-Za-z0-9]", " ", text)
    grams: set[str] = set()
    for chunk in chars.split():
        if len(chunk) >= 2:
            grams.update(chunk[i : i + 2] for i in range(len(chunk) - 1))
        else:
            grams.add(chunk)
    return grams


def _relevant(results: list[SearchResult], query: str) -> list[SearchResult]:
    """挡掉引擎的「跑偏结果」。

    两道关:
      1. 标题/摘要里出现查询词(空格分词),直接算相关;
      2. 没有空格的长问句,用中文二元组重合度判断(至少重合 2 个)。
    典型场景:Bing 在国内网络下会把「大模型 就业」答成「大」的词典释义,
    或把「large language model」答成「large」的单词解释,这类结果不能进报告。
    """
    tokens = [t.lower() for t in query.split() if len(t.strip()) >= 2]
    qgrams = _bigrams(query)
    need = 2 if len(qgrams) >= 4 else 1
    kept: list[SearchResult] = []
    for r in results:
        hay = (r.title + " " + r.snippet).lower()
        if any(t in hay for t in tokens):
            kept.append(r)
            continue
        if qgrams and sum(1 for g in qgrams if g in hay) >= need:
            kept.append(r)
    return kept


def search(query: str, limit: int = 4, timeout: int = 15, providers: list[str] | None = None) -> list[SearchResult]:
    """检索一个查询词;按引擎顺序自动降级,全都失败抛 SearchError。"""
    global _preferred
    if not query.strip():
        return []
    names = providers or _order()
    alive = [n for n in names if n not in _dead]
    names = alive or names
    errors: list[str] = []
    for name in names:
        fn = PROVIDERS.get(name)
        if not fn:
            continue
        try:
            results = fn(query, max(1, limit), timeout)
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError, SearchError) as e:
            errors.append(f"{name}: {str(e)[:70]}")
            _dead.add(name)
            continue
        results = _relevant(results, query)
        if results:
            _preferred = name  # 记住这次能用的引擎,后面的查询少走弯路
            _dead.discard(name)
            return results[:limit]
        errors.append(f"{name}: 结果与查询无关")
        _dead.add(name)
    raise SearchError("; ".join(errors) or "没有可用的检索引擎")


def search_many(
    queries: Iterable[str], limit: int = 4, timeout: int = 15
) -> tuple[list[SearchResult], list[str]]:
    """多查询检索,按 URL 去重。返回 (结果, 失败原因列表)。"""
    seen: set[str] = set()
    merged: list[SearchResult] = []
    errors: list[str] = []
    for q in queries:
        try:
            for r in search(q, limit=limit, timeout=timeout):
                if r.url in seen:
                    continue
                seen.add(r.url)
                merged.append(r)
        except SearchError as e:
            errors.append(f"{q}: {e}")
    return merged, errors


if __name__ == "__main__":  # pragma: no cover - 手工调试入口
    import sys

    q = sys.argv[1] if len(sys.argv) > 1 else "多智能体 协作 研究"
    try:
        for item in search(q, limit=5):
            print(json.dumps(item.to_dict(), ensure_ascii=False))
    except SearchError as e:
        print("检索失败:", e)
