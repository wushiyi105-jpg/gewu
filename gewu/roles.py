"""四个角色的系统提示词与调用封装。

一个角色 = 一个独立任务:只吃分配给它的输入,只产出自己那一段结果。
角色之间不共享记忆,交接全部通过结构化产物(Plan / Note / Critique / Markdown)。
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from typing import Callable

from .llm import LLMError, extract_json
from .utils import strip_md_fences

LLMFn = Callable[..., str]

ROLE_META: dict[str, dict] = {
    "planner": {"name": "规划者", "emoji": "🧭", "color": "#7aa2f7", "duty": "拆题:把课题拆成互补的子问题"},
    "researcher": {"name": "研究者", "emoji": "🔍", "color": "#9ece6a", "duty": "求证:逐个子问题检索并给出可引用发现"},
    "writer": {"name": "写作者", "emoji": "✍️", "color": "#e0af68", "duty": "成文:把发现写成有判断的文章"},
    "critic": {"name": "审稿者", "emoji": "🧐", "color": "#f7768e", "duty": "挑刺:找论据漏洞、逻辑跳跃与遗漏"},
}


# ---------------------------------------------------------------- 数据结构


@dataclass
class SubQuestion:
    id: str
    question: str
    why: str = ""
    queries: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Plan:
    angle: str
    subquestions: list[SubQuestion]

    def to_dict(self) -> dict:
        return {"angle": self.angle, "subquestions": [s.to_dict() for s in self.subquestions]}


@dataclass
class Finding:
    point: str
    detail: str
    sources: list[str] = field(default_factory=list)
    confidence: str = "model_knowledge"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Note:
    question: str
    findings: list[Finding]
    gaps: list[str] = field(default_factory=list)
    takeaway: str = ""

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "findings": [f.to_dict() for f in self.findings],
            "gaps": self.gaps,
            "takeaway": self.takeaway,
        }


@dataclass
class Critique:
    verdict: str
    strengths: list[str] = field(default_factory=list)
    issues: list[dict] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def high_issues(self) -> list[dict]:
        return [i for i in self.issues if str(i.get("severity", "")).lower() in {"high", "高", "严重"}]


# ---------------------------------------------------------------- 提示词

PLANNER_SYSTEM = """你是「格物」研究团队的**规划者**。你不写正文,只负责把一个大课题拆开。

任务:把下面的课题拆成 {depth} 个互补、可独立调研、合起来能支撑一篇深度报告的子问题。

原则:
1. 子问题之间不要重叠,要像拼图一样严丝合缝地覆盖课题;
2. 切分维度按课题性质取舍,例如:现状与事实 / 机制与原因 / 数据与规模 / 分歧与争议 / 影响与趋势 / 边界与局限;
3. 每个子问题给 1~2 个**搜索引擎查询词**——要具体、含关键实体,不要整句式问句(例:「大模型 岗位替代 研究」而不是「大模型会不会让很多人失业?」);
4. 排序:先事实,后机制,再争议与趋势。

只输出 JSON,不要解释:
{{"angle":"一句话说明你的切分角度", "subquestions":[{{"id":"Q1","question":"子问题原文","why":"为什么必须有它","queries":["查询词1","查询词2"]}}]}}"""

RESEARCHER_SYSTEM = """你是「格物」研究团队的**研究者**。你只对分配给你的一个子问题负责,产出「可以被写作者直接引用」的发现。

铁律:
- 事实性陈述只能来自【检索材料】,句末用 [S1] / [S2] 这样的编号标注来源;
- 绝不允许编造来源、URL、数字、日期、机构名;
- 材料不够时可以补常识,但该条 findings 的 confidence 必须写 `model_knowledge`,并在 detail 末尾注明「(模型知识,未核验)」;
- 宁要 3 条扎实,不要 6 条空泛。每条 detail 要带具体机制、数字或案例,不写「很重要」「影响深远」这类空话;detail 控制在 2~3 句、100 字以内,findings 最多 4 条(输出太长会被截断,截断后整段作废)。

只输出 JSON:
{{"question":"子问题原文","takeaway":"一句话回答这个问题","findings":[{{"point":"一句话结论","detail":"2~4 句展开,含具体证据","sources":["S1","S3"],"confidence":"sourced|mixed|model_knowledge"}}],"gaps":["还缺什么证据才能定论"]}}"""

WRITER_SYSTEM = """你是「格物」研究团队的**写作者**。你要把研究笔记写成一篇有判断的文章,不是把笔记拼接成段落。

结构(严格照此,标题用 Markdown):
## 摘要
120~180 字,直接给核心判断,不要「本文旨在」。

## 01 / 02 / 03 ...(按子问题分节,每节一个明确小标题)
每节开头一句判断,后面用证据支撑。引用必须用 [S3] 这种编号写在句末,只能用笔记里出现过的编号。

## 分歧与不确定性
哪些地方证据打架、哪些结论只是模型知识、哪些数字不可靠。这一节必须有,不许糊过去。

## 结论
3~5 条,每条一句话,是「所以应该怎么看/怎么办」,不是复述前文。

文风:
- 中文,1200~1800 字;
- 允许有态度的判断,但每个判断都要能落回证据;
- 禁止空话套话:不写「随着…的发展」「综上所述」「作为一个 AI」;
- 不要自己列参考资料清单(系统会统一追加真实来源)。"""

CRITIC_SYSTEM = """你是「格物」研究团队的**审稿者**。你的价值在于挑错,夸人不是你的工作。

按下面四条逐项检查:
1. 论据:有没有把「模型知识」当既定事实写?有没有 [S#] 支撑不了的断言?
2. 逻辑:因果是否跳跃?相关当因果?反例被回避了吗?
3. 覆盖:有没有漏掉重要侧面、反面观点、量级判断?
4. 结构:是不是只是三段落并列?结论有没有超出证据?

只输出 JSON:
{{"verdict":"accept|revise","strengths":["真优点,最多2条"],"issues":[{{"severity":"high|medium|low","location":"哪一节/哪一句","problem":"具体问题","fix":"怎么改"}}],"missing":["必须补上的内容"]}}
最多 6 条 issues,按严重度排序;没有真问题就写少一点,不要凑数。"""

REVISE_NOTE = """

【本次是修订,不是重写】
保留原稿骨架与已经成立的部分,优先修掉 high / medium 问题:
- 补证据:能用已有 [S#] 支撑的地方补引用;不能的改成「尚缺证据」或降级为推测;
- 拆掉没有依据的具体数字;
- 补上审稿者指出遗漏的侧面;
- 结论要往回收,别让判断大于证据。
仍然输出完整 Markdown 全文,不要输出「修改说明」。"""


# ---------------------------------------------------------------- 归一化


def _as_list(value) -> list:
    if value is None or value == "":
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        return [value]
    return [value]


def _clean_refs(items) -> list[str]:
    refs: list[str] = []
    for item in _as_list(items):
        for hit in re.findall(r"S\s*[0-9]+", str(item).upper()):
            ref = hit.replace(" ", "")
            if ref not in refs:
                refs.append(ref)
    return refs


def fallback_plan(topic: str, depth: int) -> Plan:
    """模型输出不可解析时的兜底切分(保证流程能走完)。"""
    templates = [
        f"{topic} 的现状与关键事实是什么?",
        f"驱动 {topic} 的核心机制与原因是什么?",
        f"围绕 {topic} 的主要争议与分歧在哪里?",
        f"{topic} 正在往哪里走、影响有多大?",
        f"{topic} 有哪些边界条件与不确定性?",
        f"{topic} 有哪些常被忽略的反面证据?",
    ]
    subs = [
        SubQuestion(
            id=f"Q{i + 1}",
            question=q,
            why="兜底切分",
            queries=ensure_queries(topic, q, []),
        )
        for i, q in enumerate(templates[: max(1, depth)])
    ]
    return Plan(angle="兜底切分(模型未给出可解析的拆题结果)", subquestions=subs)


FACET_RULES = [
    ("现状 数据 规模", ("现状", "事实", "数据", "规模", "多少", "实证", "采用率")),
    ("机制 原因 研究", ("机制", "原因", "为什么", "驱动", "原理", "导致", "传导", "路径")),
    ("人群 分布 差异", ("人群", "分布", "差异", "群体", "谁", "层级", "行业")),
    ("争议 反例 质疑", ("争议", "分歧", "反例", "质疑", "局限", "反对")),
    ("趋势 预测 影响", ("趋势", "走向", "未来", "影响", "前景", "判断")),
    ("边界 不确定性", ("边界", "不确定", "风险", "条件")),
]


def ensure_queries(topic: str, question: str, queries: list[str]) -> list[str]:
    """把「整句式查询词」换成搜索引擎真能用的关键词。

    模型经常把子问题原样当查询词交上来,或者给一句 20 多字的问句。
    这里做一层兜底:太长的、和问题一模一样的都丢掉,
    再用「课题 + 子问题维度(现状/机制/争议/趋势/边界)」拼一个。
    """
    good: list[str] = []
    for q in queries:
        q = str(q).strip()
        if not q or q == question.strip() or len(q) > 18:
            continue
        good.append(q)
    if good:
        return good[:2]
    best: tuple[int, str] | None = None
    for label, words in FACET_RULES:
        positions = [question.find(w) for w in words if w in question]
        if not positions:
            continue
        pos = min(positions)
        if best is None or pos < best[0]:
            best = (pos, label)
    if best:
        return [f"{topic} {best[1]}".strip()]
    return [topic or question]


def normalize_plan(raw, depth: int, topic: str = "") -> Plan:
    if isinstance(raw, list):
        items, angle = raw, ""
    elif isinstance(raw, dict):
        items = raw.get("subquestions") or raw.get("questions") or raw.get("子问题") or []
        angle = str(raw.get("angle") or raw.get("视角") or raw.get("角度") or "")
    else:
        raise LLMError("规划者返回的不是对象或数组")

    subs: list[SubQuestion] = []
    for item in _as_list(items):
        if isinstance(item, str):
            q, why, queries = item.strip(), "", []
        elif isinstance(item, dict):
            q = str(item.get("question") or item.get("q") or item.get("子问题") or "").strip()
            why = str(item.get("why") or item.get("rationale") or item.get("理由") or "").strip()
            queries = [str(x).strip() for x in _as_list(item.get("queries") or item.get("query")) if str(x).strip()]
        else:
            continue
        if not q:
            continue
        if not queries:
            queries = [q.rstrip("?？")]
        queries = ensure_queries(topic, q, queries)
        subs.append(SubQuestion(id=f"Q{len(subs) + 1}", question=q, why=why, queries=queries[:2]))
        if len(subs) >= depth:
            break

    if not subs:
        if topic:
            return fallback_plan(topic, depth)
        raise LLMError("规划者没有给出任何子问题")
    return Plan(angle=angle or "未说明切分角度", subquestions=subs)


def normalize_note(raw, question: str) -> Note:
    if isinstance(raw, str):
        raw = {"findings": [{"point": raw, "detail": raw, "confidence": "model_knowledge"}]}
    if not isinstance(raw, dict):
        raise LLMError("研究者返回的不是对象")

    findings: list[Finding] = []
    for item in _as_list(raw.get("findings") or raw.get("发现")):
        if isinstance(item, str):
            point, detail, sources, conf = item.strip(), item.strip(), [], "model_knowledge"
        elif isinstance(item, dict):
            point = str(item.get("point") or item.get("结论") or item.get("finding") or "").strip()
            detail = str(item.get("detail") or item.get("细节") or item.get("说明") or point).strip()
            sources = _clean_refs(item.get("sources") or item.get("引用") or item.get("source"))
            conf = str(item.get("confidence") or "").strip().lower() or ("sourced" if sources else "model_knowledge")
        else:
            continue
        if not point and not detail:
            continue
        findings.append(
            Finding(point=point or detail[:80], detail=detail or point, sources=sources, confidence=conf)
        )

    return Note(
        question=str(raw.get("question") or question),
        findings=findings,
        gaps=[str(g) for g in _as_list(raw.get("gaps") or raw.get("缺口")) if str(g).strip()],
        takeaway=str(raw.get("takeaway") or raw.get("结论") or "").strip(),
    )


def normalize_critique(raw) -> Critique:
    if isinstance(raw, list):
        raw = {"issues": raw}
    if not isinstance(raw, dict):
        raise LLMError("审稿者返回的不是对象")

    issues: list[dict] = []
    for item in _as_list(raw.get("issues") or raw.get("问题")):
        if isinstance(item, str):
            issues.append({"severity": "medium", "location": "", "problem": item.strip(), "fix": ""})
        elif isinstance(item, dict):
            issues.append(
                {
                    "severity": str(item.get("severity") or "medium").lower(),
                    "location": str(item.get("location") or item.get("位置") or ""),
                    "problem": str(item.get("problem") or item.get("问题") or "").strip(),
                    "fix": str(item.get("fix") or item.get("建议") or "").strip(),
                }
            )
    return Critique(
        verdict=str(raw.get("verdict") or "revise").lower(),
        strengths=[str(s) for s in _as_list(raw.get("strengths")) if str(s).strip()],
        issues=[i for i in issues if i["problem"]],
        missing=[str(m) for m in _as_list(raw.get("missing") or raw.get("遗漏")) if str(m).strip()],
    )


# ---------------------------------------------------------------- 角色调用


def plan(topic: str, depth: int, llm: LLMFn) -> tuple[Plan, str]:
    """规划者:课题 -> 子问题清单。"""
    system = PLANNER_SYSTEM.replace("{depth}", str(depth))
    messages = [{"role": "system", "content": system}, {"role": "user", "content": f"课题:{topic}"}]
    raw = llm(messages, temperature=0.4, max_tokens=1200)
    try:
        return normalize_plan(extract_json(raw), depth, topic), raw
    except LLMError:
        # 规划者的输出决定后面所有任务,解析失败不直接兜底:先按更严格的格式要求重来一次
        retry = messages + [
            {"role": "assistant", "content": raw[:2000]},
            {
                "role": "user",
                "content": "上面这段无法解析。请重新输出,严格只有一个 JSON 对象:"
                '{"angle":"...","subquestions":[{"id":"Q1","question":"...","why":"...","queries":["..."]}]},'
                "不要解释,不要 Markdown,不要代码块标记。",
            },
        ]
        raw2 = raw
        try:
            raw2 = llm(retry, temperature=0.2, max_tokens=1200)
            return normalize_plan(extract_json(raw2), depth, topic), raw2
        except LLMError:
            return fallback_plan(topic, depth), raw2


def research(question: str, evidence: str, llm: LLMFn) -> tuple[Note, str]:
    """研究者:一个子问题 + 检索材料 -> 结构化发现。"""
    user = f"子问题:{question}\n\n{evidence}\n\n请只针对这个子问题作答。"
    messages = [{"role": "system", "content": RESEARCHER_SYSTEM}, {"role": "user", "content": user}]
    # 推理型模型会把 token 花在 reasoning 上,给足预算,避免 JSON 被截断
    raw = llm(messages, temperature=0.2, max_tokens=4096)
    try:
        return normalize_note(extract_json(raw), question), raw
    except LLMError:
        retry = messages + [
            {"role": "assistant", "content": raw[:2500]},
            {
                "role": "user",
                "content": "上面的输出不是可解析的 JSON(可能被截断)。请重新输出,只要一个 JSON 对象,"
                '字段固定为 question / takeaway / findings / gaps;findings 每条含 point、detail、'
                'sources、confidence,sources 只填材料里真实出现的 S 编号。不要解释,不要代码块。',
            },
        ]
        raw2 = raw
        try:
            raw2 = llm(retry, temperature=0.1, max_tokens=4096)
            return normalize_note(extract_json(raw2), question), raw2
        except LLMError:
            return normalize_note(raw, question), raw2


def write_report(topic: str, plan_obj: Plan, notes: list[Note], llm: LLMFn) -> tuple[str, str]:
    """写作者:研究笔记 -> 报告初稿。"""
    blocks = [f"课题:{topic}", f"规划者的切分角度:{plan_obj.angle}", ""]
    for sub, note in zip(plan_obj.subquestions, notes):
        blocks.append(f"### {sub.id} {sub.question}")
        if note.takeaway:
            blocks.append(f"一句话结论:{note.takeaway}")
        for f in note.findings:
            refs = "".join(f"[{s}]" for s in f.sources) or "(无来源)"
            blocks.append(f"- {f.point} {refs} · {f.confidence}\n  {f.detail}")
        if note.gaps:
            blocks.append(f"- 证据缺口:{';'.join(note.gaps)}")
        blocks.append("")
    raw = llm(
        [{"role": "system", "content": WRITER_SYSTEM}, {"role": "user", "content": "\n".join(blocks)}],
        temperature=0.5,
        max_tokens=5000,
    )
    return strip_md_fences(raw), raw


def critique_report(topic: str, report: str, notes: list[Note], llm: LLMFn) -> tuple[Critique, str]:
    """审稿者:报告 + 原笔记 -> 结构化意见。"""
    brief = []
    for note in notes:
        refs = sorted({s for f in note.findings for s in f.sources})
        brief.append(f"- {note.question} (可用来源:{', '.join(refs) or '无'})")
    user = (
        f"课题:{topic}\n\n【报告全文】\n{report}\n\n【研究笔记里的来源编号】\n"
        + "\n".join(brief)
        + "\n\n请开始挑刺。"
    )
    raw = llm(
        [{"role": "system", "content": CRITIC_SYSTEM}, {"role": "user", "content": user}],
        temperature=0.3,
        max_tokens=2600,
    )
    try:
        return normalize_critique(extract_json(raw)), raw
    except LLMError:
        return Critique(verdict="revise", issues=[{"severity": "medium", "location": "全文", "problem": raw[:400], "fix": ""}]), raw


def revise_report(topic: str, report: str, critique: Critique, llm: LLMFn) -> tuple[str, str]:
    """写作者(修订轮):照审稿意见改稿。"""
    user = (
        f"课题:{topic}\n\n【原稿】\n{report}\n\n【审稿意见】\n"
        + json.dumps(critique.to_dict(), ensure_ascii=False, indent=2)
    )
    raw = llm(
        [{"role": "system", "content": WRITER_SYSTEM + REVISE_NOTE}, {"role": "user", "content": user}],
        temperature=0.4,
        max_tokens=5000,
    )
    return strip_md_fences(raw), raw
