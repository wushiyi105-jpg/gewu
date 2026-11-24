"""编排层:把四个角色串成一条可观测、可续跑的流水线。

一次 run 的目录结构::

    runs/20261124-195500/
    ├── meta.json          # 课题、模型、参数、耗时
    ├── tasks.json         # 任务看板(每个角色一次调用 = 一条独立任务)
    ├── cache.json         # 断点续跑用的任务缓存(按 角色+输入 哈希)
    ├── tasks/
    │   ├── T1_planner.json / .md
    │   ├── T2_researcher_Q1.json / .md
    │   ├── ...
    │   ├── T6_writer_draft.md
    │   ├── T7_critic.json / .md
    │   └── T8_writer_revise.md
    └── report.md          # 终稿(含统一追加的来源清单与任务看板)
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Callable

from . import roles, search as searchmod
from .llm import LLMError, chat, default_config
from .utils import count_words, now_str, run_stamp, truncate

RUNS_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runs")


def runs_root_path() -> str:
    """run 的落盘根目录,可用 GEWU_RUNS_ROOT 覆盖(测试与多项目隔离用)。"""
    return os.environ.get("GEWU_RUNS_ROOT") or RUNS_ROOT


@dataclass
class TaskRecord:
    id: str
    role: str
    title: str
    status: str = "pending"  # pending / running / done / reused / failed / skipped
    started: str = ""
    elapsed: float = 0.0
    summary: str = ""
    artifact: str = ""
    error: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def role_name(self) -> str:
        return roles.ROLE_META.get(self.role, {}).get("name", self.role)


@dataclass
class Source:
    sid: str
    title: str
    url: str
    snippet: str = ""
    query: str = ""
    question: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class RunResult:
    run_id: str
    run_dir: str
    report: str
    report_path: str
    tasks: list[TaskRecord]
    plan: roles.Plan | None = None
    notes: list[roles.Note] = field(default_factory=list)
    critique: roles.Critique | None = None
    sources: list[Source] = field(default_factory=list)
    elapsed: float = 0.0
    error: str = ""

    @property
    def ok(self) -> bool:
        return bool(self.report) and not self.error


class Orchestrator:
    """规划 → 研究(×N) → 成文 → 批判 → 修订 的编排循环。"""

    def __init__(
        self,
        topic: str,
        *,
        depth: int = 4,
        use_search: bool = True,
        llm: Callable[..., str] | None = None,
        on_event: Callable[[dict], None] | None = None,
        runs_root: str | None = None,
        run_id: str | None = None,
        resume_dir: str | None = None,
        search_limit: int = 4,
        search_fn: Callable[..., tuple[list, list]] | None = None,
        revise: bool = True,
        model_label: str | None = None,
    ):
        self.topic = topic.strip()
        self.depth = max(1, min(6, depth))
        self.use_search = use_search
        self.llm = llm or (lambda messages, **kw: chat(messages, config=self.config, **kw))
        self.on_event = on_event
        self.runs_root = runs_root or runs_root_path()
        self.search_limit = search_limit
        self.search_fn = search_fn or searchmod.search_many
        self.revise = revise
        self.config = default_config()
        self.model_label = model_label or self.config.get("model") or "unknown"

        self.resume_dir = resume_dir
        self.run_id = run_id or run_stamp()
        self.run_dir = resume_dir or os.path.join(self.runs_root, self.run_id)

        self.tasks: list[TaskRecord] = []
        self.sources: list[Source] = []
        self.cache: dict[str, dict] = {}
        self._counter = 0
        self._t0 = time.time()

    # ------------------------------------------------------------ 事件与目录

    def _emit(self, kind: str, **payload):
        if not self.on_event:
            return
        try:
            self.on_event({"type": kind, "run_id": self.run_id, "ts": now_str("%H:%M:%S"), **payload})
        except Exception:  # 前端/终端回调出错不能拖垮主流程
            pass

    def _log(self, task: TaskRecord, text: str):
        self._emit("task_log", task_id=task.id, role=task.role, text=text)

    def _ensure_dirs(self):
        os.makedirs(os.path.join(self.run_dir, "tasks"), exist_ok=True)

    def _next_id(self) -> str:
        self._counter += 1
        return f"T{self._counter}"

    # ------------------------------------------------------------ 缓存(独立任务)

    def _cache_path(self) -> str:
        return os.path.join(self.run_dir, "cache.json")

    def _load_cache(self):
        path = self._cache_path()
        if os.path.isfile(path):
            try:
                with open(path, encoding="utf-8") as f:
                    self.cache = json.load(f) or {}
            except (json.JSONDecodeError, OSError):
                self.cache = {}

    def _save_cache(self):
        try:
            with open(self._cache_path(), "w", encoding="utf-8") as f:
                json.dump(self.cache, f, ensure_ascii=False, indent=2)
        except OSError:
            pass

    @staticmethod
    def cache_key(role: str, material: str) -> str:
        return hashlib.sha1(f"{role}::{material}".encode("utf-8")).hexdigest()

    # ------------------------------------------------------------ 任务生命周期

    def _placeholder(self, role: str, title: str) -> TaskRecord:
        return TaskRecord(id=self._next_id(), role=role, title=title)

    def _start(self, task: TaskRecord):
        task.status = "running"
        task.started = now_str("%H:%M:%S")
        task._t0 = time.time()  # type: ignore[attr-defined]
        self.tasks.append(task)
        self._emit("task_start", task=task.to_dict())

    def _finish(self, task: TaskRecord, status: str, summary: str, artifact: str = ""):
        task.status = status
        task.elapsed = round(time.time() - getattr(task, "_t0", time.time()), 1)
        task.summary = summary
        task.artifact = artifact
        self._emit("task_end", task=task.to_dict(), summary=summary, artifact=artifact)
        self._write_tasks()

    def _fail(self, task: TaskRecord, error: str):
        task.status = "failed"
        task.error = error
        task.summary = truncate(error, 120)
        self._emit("task_end", task=task.to_dict(), summary=task.summary, artifact="", error=error)
        self._write_tasks()

    def _wrap_up(self, task: TaskRecord, summary: str, artifact: str = ""):
        """失败的兜底也要收尾:状态保持 failed,别被后一次 _finish 覆盖成 done。"""
        if task.status == "failed":
            self._finish(task, "failed", task.summary, artifact)
        else:
            self._finish(task, "done", summary, artifact)

    def _write_tasks(self):
        try:
            with open(os.path.join(self.run_dir, "tasks.json"), "w", encoding="utf-8") as f:
                json.dump([t.to_dict() for t in self.tasks], f, ensure_ascii=False, indent=2)
        except OSError:
            pass

    def _save_json(self, task_id: str, role: str, data: dict, raw: str = "") -> str:
        path = os.path.join("tasks", f"{task_id}_{role}.json")
        full = os.path.join(self.run_dir, path)
        payload = {"task_id": task_id, "role": role, "data": data, "raw": raw}
        with open(full, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        return path.replace("\\", "/")

    def _save_md(self, task_id: str, role: str, markdown: str) -> str:
        path = os.path.join("tasks", f"{task_id}_{role}.md")
        with open(os.path.join(self.run_dir, path), "w", encoding="utf-8") as f:
            f.write(markdown)
        return path.replace("\\", "/")

    # ------------------------------------------------------------ 主流程

    def run(self) -> RunResult:
        self._ensure_dirs()
        self._load_cache()
        self._emit(
            "run_start",
            topic=self.topic,
            depth=self.depth,
            search=self.use_search,
            model=self.model_label,
            run_dir=self.run_dir,
        )

        plan_obj = self._stage_plan()
        notes = self._stage_research(plan_obj)
        draft = self._stage_writer(plan_obj, notes)
        critique = self._stage_critic(draft, notes)
        final = self._stage_revise(draft, critique, notes)
        report = self._compose(final or draft, plan_obj, notes, critique)
        report_path = self._save_report(report)

        elapsed = round(time.time() - self._t0, 1)
        self._write_meta(elapsed, report)
        self._emit("final", report_path=report_path, chars=count_words(report))
        self._emit("run_end", elapsed=elapsed, report_path=report_path, tasks=len(self.tasks))

        return RunResult(
            run_id=self.run_id,
            run_dir=self.run_dir,
            report=report,
            report_path=report_path,
            tasks=self.tasks,
            plan=plan_obj,
            notes=notes,
            critique=critique,
            sources=self.sources,
            elapsed=elapsed,
        )

    # -------- ① 规划者

    def _stage_plan(self) -> roles.Plan:
        task = self._placeholder("planner", "拆解课题")
        src = self.cache_key("planner", f"{self.topic}|{self.depth}")
        hit = self.cache.get(src)
        if hit:
            task.id = hit.get("task_id", task.id)
            task.status = "reused"
            task.artifact = hit.get("artifact") or self._find_artifact(task.id)
            task.summary = "复用上次拆题结果"
            self.tasks.append(task)
            plan_obj = roles.normalize_plan(hit.get("data") or {}, self.depth, self.topic)
            self._emit("task_end", task=task.to_dict(), summary=task.summary, artifact="", reused=True)
            self._write_tasks()
            return plan_obj

        self._start(task)
        try:
            plan_obj, raw = roles.plan(self.topic, self.depth, self.llm)
        except LLMError as e:
            self._fail(task, f"规划失败:{e}")
            plan_obj, raw = roles.fallback_plan(self.topic, self.depth), ""
            self._emit("task_log", task_id=task.id, role=task.role, text="⚠️ 走向兜底切分")

        for sub in plan_obj.subquestions:
            self._log(task, f"{sub.id} {sub.question}")
        if plan_obj.angle.startswith("兜底"):
            self._emit(
                "task_log",
                task_id=task.id,
                role=task.role,
                text="⚠️ 规划者两次输出都无法解析,已用兜底切分(原文见任务产物)",
            )
        artifact = self._save_json(task.id, "planner", plan_obj.to_dict(), raw)
        md = ["# 拆题结果", "", f"切分角度:{plan_obj.angle}", ""]
        md += [f"- **{s.id}** {s.question}\n  - 为何需要:{s.why}\n  - 查询词:{' / '.join(s.queries)}" for s in plan_obj.subquestions]
        self._save_md(task.id, "planner", "\n".join(md))
        self.cache[src] = {"task_id": task.id, "data": plan_obj.to_dict(), "raw": raw, "artifact": artifact}
        self._save_cache()
        self._wrap_up(task, f"{len(plan_obj.subquestions)} 个子问题", artifact)
        return plan_obj

    # -------- ② 研究者(×N,每个子问题一条独立任务)

    def _stage_research(self, plan_obj: roles.Plan) -> list[roles.Note]:
        notes: list[roles.Note] = []
        for sub in plan_obj.subquestions:
            notes.append(self._research_one(sub))
        return notes

    def _research_one(self, sub: roles.SubQuestion) -> roles.Note:
        task = self._placeholder("researcher", f"{sub.id} · {truncate(sub.question, 32)}")
        material = f"{sub.id}|{sub.question}|{','.join(sub.queries)}|search={self.use_search}"
        src = self.cache_key("researcher", material)
        hit = self.cache.get(src)
        if hit:
            task.id = hit.get("task_id", task.id)
            task.status = "reused"
            task.artifact = hit.get("artifact") or self._find_artifact(task.id)
            task.summary = "复用上次研究结果"
            self.tasks.append(task)
            note = roles.normalize_note(hit.get("data") or {}, sub.question)
            for s in hit.get("sources") or []:
                if isinstance(s, dict) and s.get("url"):
                    self.sources.append(Source(**{k: s.get(k, "") for k in ("sid", "title", "url", "snippet", "question", "query")}))
            self._emit("task_end", task=task.to_dict(), summary=task.summary, artifact="", reused=True)
            self._write_tasks()
            return note

        self._start(task)
        found: list[searchmod.SearchResult] = []
        search_errors: list[str] = []
        if self.use_search:
            self._log(task, f"检索:{' / '.join(sub.queries)}")
            try:
                found, search_errors = self.search_fn(sub.queries, limit=self.search_limit)
            except Exception as e:  # 检索层任何异常都只降级,不中断
                search_errors = [str(e)]
            for err in search_errors:
                self._log(task, f"⚠️ 检索降级:{truncate(err, 80)}")
            for r in found:
                self._log(task, f"命中:{truncate(r.title, 46)}")

        evidence, new_sources = self._build_evidence(found, sub)
        self.sources.extend(new_sources)

        if not new_sources and self.use_search:
            evidence += "\n(本次检索没有拿到可用材料,只能凭模型知识作答,请严格标注 model_knowledge)"

        try:
            note, raw = roles.research(sub.question, evidence, self.llm)
        except LLMError as e:
            self._fail(task, f"研究失败:{e}")
            note, raw = roles.Note(
                question=sub.question,
                findings=[],
                gaps=[f"模型调用失败:{e}"],
                takeaway="该子问题未能完成研究",
            ), ""

        artifact = self._save_json(task.id, "researcher", note.to_dict(), raw)
        self._save_md(task.id, "researcher", self._note_markdown(sub, note))
        self.cache[src] = {
            "task_id": task.id,
            "data": note.to_dict(),
            "raw": raw,
            "artifact": artifact,
            "sources": [s.to_dict() for s in new_sources],
        }
        self._save_cache()
        sourced = sum(1 for f in note.findings if f.sources)
        self._wrap_up(task, f"{len(note.findings)} 条发现 · {sourced} 条有来源", artifact)
        return note

    def _build_evidence(self, found: list[searchmod.SearchResult], sub: roles.SubQuestion):
        """给检索结果编号并生成材料正文。只返回新来源,由调用方统一登记,避免重复入库。"""
        new_sources: list[Source] = []
        lines = ["【检索材料】"]
        if not found:
            lines.append("(无联网材料)")
        base = len(self.sources)
        for i, r in enumerate(found):
            sid = f"S{base + i + 1}"
            src = Source(sid=sid, title=r.title, url=r.url, snippet=r.snippet, question=sub.id, query="")
            new_sources.append(src)
            lines.append(f"[{sid}] {r.title}\n    URL: {r.url}\n    摘要: {truncate(r.snippet, 220)}")
        return "\n".join(lines), new_sources

    # -------- ③ 写作者(初稿)

    def _stage_writer(self, plan_obj: roles.Plan, notes: list[roles.Note]) -> str:
        task = self._placeholder("writer", "整合成文 · 初稿")
        src = self.cache_key("writer", f"{self.topic}|" + json.dumps([n.to_dict() for n in notes], ensure_ascii=False, sort_keys=True))
        hit = self.cache.get(src)
        if hit:
            task.id = hit.get("task_id", task.id)
            task.status = "reused"
            task.artifact = hit.get("artifact") or self._find_artifact(task.id)
            task.summary = "复用上次初稿"
            self.tasks.append(task)
            self._emit("task_end", task=task.to_dict(), summary=task.summary, artifact="", reused=True)
            self._write_tasks()
            return hit.get("data", {}).get("draft", "")

        self._start(task)
        self._log(task, f"输入:{len(notes)} 份研究笔记")
        try:
            draft, raw = roles.write_report(self.topic, plan_obj, notes, self.llm)
        except LLMError as e:
            self._fail(task, f"写作失败:{e}")
            draft, raw = self.fallback_body(plan_obj, notes), ""

        draft = self._ensure_sections(draft, notes)
        artifact = self._save_md(task.id, "writer", draft)
        self.cache[src] = {"task_id": task.id, "data": {"draft": draft}, "raw": raw, "artifact": artifact}
        self._save_cache()
        self._wrap_up(task, f"{count_words(draft)} 字", artifact)
        return draft

    def _ensure_sections(self, body: str, notes: list[roles.Note]) -> str:
        """模型偶尔漏写「分歧与不确定性」「结论」:用笔记里的现成内容补齐,保证报告结构完整。"""
        text = body.strip()
        if "## 摘要" not in text:
            lead = "。".join(n.takeaway.rstrip("。") for n in notes if n.takeaway)[:220]
            text = f"## 摘要\n\n{lead}。\n\n{text}" if lead else text
        if "## 分歧与不确定性" not in text:
            gaps = [g for n in notes for g in n.gaps][:6]
            bullets = [f"- {g}" for g in gaps] or ["- 检索材料有限,以下判断多为模型知识,未做外部核验。"]
            text += "\n\n## 分歧与不确定性\n\n" + "\n".join(bullets)
        if "## 结论" not in text:
            items = [f"- {n.takeaway}" for n in notes if n.takeaway][:5]
            text += "\n\n## 结论\n\n" + ("\n".join(items) or "- 证据不足,暂不下结论。")
        return text

    # -------- ④ 审稿者

    def _stage_critic(self, draft: str, notes: list[roles.Note]) -> roles.Critique:
        task = self._placeholder("critic", "挑刺:论据 / 逻辑 / 遗漏")
        src = self.cache_key("critic", f"{self.topic}|{draft}")
        hit = self.cache.get(src)
        if hit:
            task.id = hit.get("task_id", task.id)
            task.status = "reused"
            task.artifact = hit.get("artifact") or self._find_artifact(task.id)
            task.summary = "复用上次审稿意见"
            self.tasks.append(task)
            critique = roles.normalize_critique(hit.get("data") or {})
            self._emit("task_end", task=task.to_dict(), summary="复用上次审稿意见", artifact="", reused=True)
            self._write_tasks()
            return critique

        self._start(task)
        self._log(task, "检查论据是否可回溯、因果是否跳跃、是否漏掉反例")
        try:
            critique, raw = roles.critique_report(self.topic, draft, notes, self.llm)
        except LLMError as e:
            self._fail(task, f"审稿失败:{e}")
            critique, raw = roles.Critique(verdict="revise", issues=[]), ""

        artifact = self._save_json(task.id, "critic", critique.to_dict(), raw)
        self._save_md(task.id, "critic", self._critique_markdown(critique))
        self.cache[src] = {"task_id": task.id, "data": critique.to_dict(), "raw": raw, "artifact": artifact}
        self._save_cache()
        high = len(critique.high_issues)
        self._wrap_up(task, f"verdict={critique.verdict} · {len(critique.issues)} 条意见({high} 高)", artifact)
        return critique

    # -------- ⑤ 写作者(修订)

    def _stage_revise(self, draft: str, critique: roles.Critique, notes: list[roles.Note]) -> str:
        if not self.revise or not critique.issues:
            return ""
        task = self._placeholder("writer", "按审稿意见修订")
        src = self.cache_key("revise", f"{draft}|{json.dumps(critique.to_dict(), ensure_ascii=False, sort_keys=True)}")
        hit = self.cache.get(src)
        if hit:
            task.id = hit.get("task_id", task.id)
            task.status = "reused"
            task.artifact = hit.get("artifact") or self._find_artifact(task.id)
            task.summary = "复用上次修订稿"
            self.tasks.append(task)
            self._emit("task_end", task=task.to_dict(), summary=task.summary, artifact="", reused=True)
            self._write_tasks()
            return hit.get("data", {}).get("draft", "")

        self._start(task)
        self._log(task, f"按 {len(critique.issues)} 条意见改稿")
        try:
            final, raw = roles.revise_report(self.topic, draft, critique, self.llm)
        except LLMError as e:
            self._fail(task, f"修订失败:{e}")
            return ""

        # 修订稿比初稿短一大截、或把结论写丢了,说明模型没按「改稿」的要求做:
        # 宁可用初稿,也不要把一篇残缺的稿子交给用户
        short = count_words(final) < 0.6 * max(1, count_words(draft))
        if short or "## 结论" not in final:
            self._log(task, "⚠️ 修订稿过短或缺结论,保留初稿")
            self._finish(task, "done", "修订未通过体量检查,保留初稿", "")
            return ""

        final = self._ensure_sections(final, notes)
        artifact = self._save_md(task.id, "writer", final)
        self.cache[src] = {"task_id": task.id, "data": {"draft": final}, "raw": raw, "artifact": artifact}
        self._save_cache()
        self._finish(task, "done", f"{count_words(final)} 字", artifact)
        return final

    # ------------------------------------------------------------ 组装报告

    def _find_artifact(self, task_id: str) -> str:
        """续跑时凭 task_id 找回上次的产物文件,任务看板上的链接不至于变成「—」。"""
        folder = os.path.join(self.run_dir, "tasks")
        if not os.path.isdir(folder):
            return ""
        for name in sorted(os.listdir(folder)):
            if name.startswith(f"{task_id}_"):
                return os.path.join("tasks", name)
        return ""

    def _compose(self, body: str, plan_obj: roles.Plan, notes: list[roles.Note], critique: roles.Critique) -> str:
        mode = f"联网检索 {len(self.sources)} 条来源" if self.use_search else "纯模型知识"
        meta = (
            f"> {len(plan_obj.subquestions)} 个子问题 · {mode} · 模型 `{self.model_label}` · "
            f"{now_str()} · run `{self.run_id}`"
        )
        parts = [
            f"# {self.topic}", "", meta, "", body.strip(), "",
            self._sources_section(), "", self._task_board(),
        ]
        return "\n".join(parts).strip() + "\n"

    def _sources_section(self) -> str:
        if not self.sources:
            return (
                "## 参考资料\n\n"
                "> 本次未联网或检索无结果,报告依据模型自身知识,未做外部核验。\n"
            )
        lines = ["## 参考资料", ""]
        for s in self.sources:
            snippet = f" — {truncate(s.snippet, 90)}" if s.snippet else ""
            lines.append(f"{s.sid[1:]}. [{s.title}]({s.url}){snippet}")
        return "\n".join(lines)

    def _task_board(self) -> str:
        rows = ["## 附录 · 协作任务看板", "", "| 任务 | 角色 | 任务内容 | 状态 | 耗时 | 产物 |", "|---|---|---|---|---|---|"]
        status_cn = {"done": "✅ 完成", "reused": "♻️ 复用", "failed": "⚠️ 失败", "skipped": "⏭ 跳过", "running": "…"}
        for t in self.tasks:
            art = f"[{os.path.basename(t.artifact)}]({t.artifact})" if t.artifact else "—"
            secs = f"{t.elapsed:.1f}s" if t.elapsed else "0.0s"
            rows.append(
                f"| {t.id} | {roles.ROLE_META.get(t.role, {}).get('emoji', '')} {t.role_name} | "
                f"{t.title} | {status_cn.get(t.status, t.status)} | {secs} | {art} |"
            )
        return "\n".join(rows)

    def _note_markdown(self, sub: roles.SubQuestion, note: roles.Note) -> str:
        md = [f"# {sub.id} {sub.question}", "", f"- 查询词:{' / '.join(sub.queries)}", f"- 一句话结论:{note.takeaway or '(空)'}", ""]
        for f in note.findings:
            refs = " ".join(f"[{s}]" for s in f.sources) or "(无来源)"
            md += [f"## {f.point}", "", f"{f.detail}  {refs}  `{f.confidence}`", ""]
        if note.gaps:
            md += ["## 证据缺口", ""] + [f"- {g}" for g in note.gaps]
        return "\n".join(md)

    def _critique_markdown(self, critique: roles.Critique) -> str:
        md = [f"# 审稿意见(verdict={critique.verdict})", ""]
        if critique.strengths:
            md += ["## 成立的部分", ""] + [f"- {s}" for s in critique.strengths] + [""]
        md += ["## 问题", ""]
        for i, issue in enumerate(critique.issues, 1):
            md.append(f"{i}. **[{issue['severity']}]** {issue['location']}\n   - 问题:{issue['problem']}\n   - 改法:{issue['fix']}")
        if critique.missing:
            md += ["", "## 必须补上", ""] + [f"- {m}" for m in critique.missing]
        return "\n".join(md)

    def fallback_body(self, plan_obj: roles.Plan, notes: list[roles.Note]) -> str:
        """写作者掉线时的兜底正文:直接把笔记整理成有结构的文章,不做二次创作。"""
        md = ["## 摘要", "", f"围绕「{self.topic}」,本次研究拆出 {len(notes)} 个子问题,以下为逐项结论(模型写作环节失败,此为笔记直出)。", ""]
        for i, (sub, note) in enumerate(zip(plan_obj.subquestions, notes), 1):
            md += [f"## {i:02d} {sub.question}", ""]
            if note.takeaway:
                md += [f"**判断:**{note.takeaway}", ""]
            for f in note.findings:
                refs = "".join(f"[{s}]" for s in f.sources)
                md.append(f"- {f.point}{(' ' + refs) if refs else ''}:{f.detail}")
            md.append("")
        md += ["## 分歧与不确定性", "", "本稿由笔记直出,未做交叉论证;标 `model_knowledge` 的部分未核验。", "", "## 结论", "", "- 详见各节判断。"]
        return "\n".join(md)

    # ------------------------------------------------------------ 落盘

    def _save_report(self, report: str) -> str:
        path = os.path.join(self.run_dir, "report.md")
        with open(path, "w", encoding="utf-8") as f:
            f.write(report)
        return path

    def _write_meta(self, elapsed: float, report: str):
        payload = {
            "run_id": self.run_id,
            "topic": self.topic,
            "depth": self.depth,
            "search": self.use_search,
            "model": self.model_label,
            "finished_at": now_str(),
            "elapsed": elapsed,
            "sources": len(self.sources),
            "chars": count_words(report),
            "tasks": [t.to_dict() for t in self.tasks],
        }
        try:
            with open(os.path.join(self.run_dir, "meta.json"), "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
        except OSError:
            pass


def list_runs(runs_root: str | None = None) -> list[dict]:
    """列出历史 run(按时间倒序),给 `python -m gewu runs` 和 Web 首页用。"""
    root = runs_root or runs_root_path()
    if not os.path.isdir(root):
        return []
    runs = []
    for name in sorted(os.listdir(root), reverse=True):
        meta_path = os.path.join(root, name, "meta.json")
        if not os.path.isfile(meta_path):
            continue
        try:
            with open(meta_path, encoding="utf-8") as f:
                meta = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        meta["dir"] = os.path.join(root, name)
        runs.append(meta)
    return runs


