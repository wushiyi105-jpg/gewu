"""零碎小工具(Markdown 清洗、字数统计、时间戳)。"""

from __future__ import annotations

import re
from datetime import datetime


def strip_md_fences(text: str) -> str:
    """去掉模型有时自作多情加的 ```markdown 包裹。"""
    t = (text or "").strip()
    m = re.match(r"^```[a-zA-Z]*\s*\n(.*?)\n?```$", t, re.DOTALL)
    if m:
        return m.group(1).strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z]*\s*\n?", "", t)
        t = re.sub(r"\n?```\s*$", "", t)
    return t.strip()


def count_words(text: str) -> int:
    """粗算中文字数:汉字 + 英文单词。"""
    cjk = len(re.findall(r"[\u4e00-\u9fff]", text or ""))
    latin = len(re.findall(r"[A-Za-z]+", text or ""))
    return cjk + latin


def truncate(text: str, limit: int, suffix: str = "…") -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= limit else text[: limit - len(suffix)] + suffix


def now_str(fmt: str = "%Y-%m-%d %H:%M:%S") -> str:
    return datetime.now().strftime(fmt)


def run_stamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def md_headings(text: str) -> list[str]:
    return [ln.strip() for ln in (text or "").splitlines() if ln.strip().startswith("#")]
