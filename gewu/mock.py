"""离线演示引擎:不需要 API key、不需要网络,也能把整条链路跑完。

它按角色返回与真实模型同构的 JSON / Markdown,用于:
  - `python -m gewu --mock` 零配置演示;
  - 单元测试(不消耗 token、不受网络影响);
  - 录制演示 GIF。

注意:这里的内容是模板化的,只保证「格式与链路正确」,不代表真实研究结论。
"""

from __future__ import annotations

import json
import re


def _user_text(messages: list[dict]) -> str:
    return "\n".join(m.get("content", "") for m in messages if m.get("role") == "user")


def _system_text(messages: list[dict]) -> str:
    return "\n".join(m.get("content", "") for m in messages if m.get("role") == "system")


def _topic(user: str) -> str:
    m = re.search(r"课题:([^\n]+)", user)
    return m.group(1).strip() if m else "一个课题"


def _subquestion(user: str) -> str:
    m = re.search(r"子问题:([^\n]+)", user)
    return m.group(1).strip() if m else "子问题"


def _note_blocks(user: str) -> list[tuple[str, str]]:
    """从输入里取出 (子问题, 该节笔记文本)。"""
    blocks = re.findall(r"### (Q\d+ .+?)\n(.*?)(?=\n### |\n课题|$)", user, re.DOTALL)
    return [(b[0].strip(), b[1].strip()) for b in blocks]


def _refs(text: str) -> list[str]:
    return [f"S{n}" for n in sorted({int(x) for x in re.findall(r"\[S(\d+)\]", text)})]


def _mock_plan(user: str) -> str:
    topic = _topic(user)
    subs = [
        {
            "id": "Q1",
            "question": f"{topic}:目前的现实状况和可核实的数字是什么?",
            "why": "先把事实层钉住,后面的机制讨论才不会飘",
            "queries": [f"{topic} 现状 数据", f"{topic} 规模 统计"],
        },
        {
            "id": "Q2",
            "question": f"驱动 {topic} 的核心机制是什么?",
            "why": "解释为什么会这样,而不是只描述表象",
            "queries": [f"{topic} 机制 原因", f"{topic} 原理 研究"],
        },
        {
            "id": "Q3",
            "question": f"围绕 {topic} 的主要争议和反例在哪里?",
            "why": "把最强反方观点摆上桌,结论才站得住",
            "queries": [f"{topic} 争议 反例", f"{topic} 局限 质疑"],
        },
        {
            "id": "Q4",
            "question": f"{topic} 的走向和影响该如何判断?",
            "why": "研究最终要落到趋势判断与行动含义",
            "queries": [f"{topic} 趋势 预测", f"{topic} 影响 评估"],
        },
    ]
    return json.dumps({"angle": "事实 → 机制 → 争议 → 趋势(由硬到软递进)", "subquestions": subs[:4]}, ensure_ascii=False)


def _mock_research(user: str) -> str:
    question = _subquestion(user)
    refs = _refs(user)
    sourced = bool(refs)
    findings = [
        {
            "point": "规模与增速都存在,但口径差异很大",
            "detail": "不同统计口径给出的规模数字相差约 2~3 倍,通常是因为统计边界(是否计入间接相关岗位)不同。引用时最好同时给出数字和口径。"
            + ("" if sourced else "(模型知识,未核验)"),
            "sources": refs[:2],
            "confidence": "sourced" if sourced else "model_knowledge",
        },
        {
            "point": "真正的差异来自任务结构,而不是行业标签",
            "detail": "同一个岗位里,重复性任务占比高的部分先被压缩,需要判断、协调、承担责任的环节反而更值钱。这也解释了为什么总量数据和个人体感经常对不上。"
            + ("" if sourced else "(模型知识,未核验)"),
            "sources": refs[2:4],
            "confidence": "mixed" if sourced else "model_knowledge",
        },
        {
            "point": "时间尺度决定结论",
            "detail": "1~2 年内看到的多是任务重组;5~10 年才可能出现岗位结构的整体位移。把短期摩擦当成长期终局,是最常见的误判。"
            + ("" if sourced else "(模型知识,未核验)"),
            "sources": refs[4:5],
            "confidence": "model_knowledge",
        },
    ]
    return json.dumps(
        {
            "question": question,
            "takeaway": "结论要分清「任务被替代」和「岗位被消灭」,两者节奏完全不同。",
            "findings": findings,
            "gaps": ["缺少同口径的长周期纵向数据", "反例样本(未受影响的岗位)覆盖不足"],
        },
        ensure_ascii=False,
    )


def _mock_draft(user: str) -> str:
    topic = _topic(user)
    blocks = _note_blocks(user)
    refs = _refs(user) or ["S1"]
    body = [f"## 摘要", "", f"「{topic}」这类问题,最容易犯的错是把「任务被自动化」直接等同于「岗位被消灭」。本次研究拆出 {len(blocks) or 3} 个子问题后,更站得住的判断是:**冲击先发生在任务层,再慢慢传导到岗位层**;总量数据和个人体感之所以长期对不上,是因为统计口径与时间尺度不同。[{refs[0]}]", ""]
    sections = [
        (
            "先把事实层钉住,才能谈机制。现有材料的规模数字口径不一,直接比较会得出互相矛盾的结论。",
            "不同来源对规模的估算差了数倍,原因不在事实本身,而在统计边界:是否把间接相关岗位、外包与平台零工计入。引用任何数字,都应该连同口径一起引用。",
        ),
        (
            "决定影响大小的不是行业标签,而是任务结构。",
            "同一岗位内,重复性高的任务最先被压缩;需要判断、协调、承担责任的环节反而被放大。这解释了为什么宏观岗位数没有骤降,个体却明显感到「活变了」。",
        ),
        (
            "反方证据主要来自时间尺度与替代成本两端。",
            "把 1~2 年的任务重组外推成 5~10 年的岗位消失,是这类讨论最常见的越界。替代成本、监管约束与组织惯性都会显著拖慢节奏。",
        ),
        (
            "趋势判断要拆成两条线:技术渗透速度,和组织吸收能力。",
            "技术可用不等于马上落地——采购、培训、责任归属都会拖慢渗透;而吸收能力强的环节,先出现的是「人机配比」变化,不是裁员。两条线不同步,总量预测就很难准。",
        ),
    ]
    for i, (title, text) in enumerate(blocks, 1):
        head = title.split(" ", 1)[1] if " " in title else title
        judge, detail = sections[min(i - 1, len(sections) - 1)]
        body += [f"## {i:02d} {head}", "", f"**判断:** {judge}", "", f"{detail}[{refs[min(i - 1, len(refs) - 1)]}]", ""]
    body += [
        "## 分歧与不确定性",
        "",
        "- **口径之争**:规模数字的差异多来自统计边界,而非事实冲突;",
        "- **时间尺度**:短期摩擦与长期结构位移被混在一起讨论;",
        "- **未核验部分**:任务结构的判断目前更多是机制推理,缺少同口径纵向数据支撑。",
        "",
        "## 结论",
        "",
        "1. 先把「任务被替代」与「岗位被消灭」分开说,否则讨论必然失焦;",
        "2. 判断影响时看任务结构,不要只看行业标签;",
        "3. 任何规模数字都要带上口径与时间尺度;",
        "4. 对个人而言,可迁移的判断力比具体工具熟练度更保值。",
    ]
    return "\n".join(body)


def _mock_revision(user: str) -> str:
    draft = user.split("【原稿】", 1)[1].split("【审稿意见】", 1)[0].strip() if "【原稿】" in user else _mock_draft(user)
    lines = draft.splitlines()
    patched = []
    for ln in lines:
        if "之所以长期对不上" in ln:
            patched.append(ln.replace("之所以长期对不上,是因为统计口径与时间尺度不同", "之所以长期对不上,根子在口径:同一件事被统计记成了两种东西"))
            continue
        patched.append(ln)
    extra = [
        "",
        "## 补充:反例与边界",
        "",
        "把结论往回拉一档:在替代成本高、责任难以外包的环节,变化明显更慢;而在标准化程度高、结果可即时校验的环节,压缩会更快发生。这条边界比任何总量预测都更有解释力。",
    ]
    text = "\n".join(patched)
    if "## 补充:反例与边界" not in text:
        text = text.replace("## 结论", "\n".join(extra) + "\n\n## 结论")
    return text


def _mock_critique(user: str) -> str:
    return json.dumps(
        {
            "verdict": "revise",
            "strengths": ["判断先行、证据在后,读起来不散", "承认了口径差异,没有硬凑单一数字"],
            "issues": [
                {
                    "severity": "high",
                    "location": "摘要",
                    "problem": "「冲击先发生在任务层」这句是全篇支点,但只有一处引用,而且该来源只覆盖了一个行业。",
                    "fix": "补第二条独立来源,或把断言降级为「在已观察到的样本中」。",
                },
                {
                    "severity": "medium",
                    "location": "02 节",
                    "problem": "任务结构解释了「体感」,但没有给出任何可验证的观察口径,读者无法自行判断真假。",
                    "fix": "补一个可操作的口径,例如「同一岗位内重复性任务占比」如何测量。",
                },
                {
                    "severity": "medium",
                    "location": "全文",
                    "problem": "反方证据只用一段话带过,缺少「为什么反方可能对」的正面陈述。",
                    "fix": "单列一节,把最强反方观点完整讲一遍再回应。",
                },
            ],
            "missing": ["替代成本与责任归属的边界讨论"],
        },
        ensure_ascii=False,
    )


def make_mock_llm():
    """返回一个与真实 LLM 同签名的假模型。"""

    def mock_llm(messages, **kwargs) -> str:
        system = _system_text(messages)
        user = _user_text(messages)
        if "规划者" in system:
            return _mock_plan(user)
        if "研究者" in system:
            return _mock_research(user)
        # 注意顺序:修订提示词里会提到「审稿者」,写作者分支必须先判
        if "写作者" in system:
            if "修订" in system:
                return _mock_revision(user)
            return _mock_draft(user)
        if "审稿者" in system:
            return _mock_critique(user)
        return "（离线演示）这是一段占位文本。"

    return mock_llm
