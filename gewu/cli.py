"""命令行入口:一个课题进,一份报告出,过程全在终端里可见。"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys

from . import __version__, mock, roles
from .llm import LLMError, chat, config_from_args
from .orchestrator import Orchestrator, list_runs, runs_root_path
from .utils import count_words, truncate

C = {
    "dim": "\033[2m",
    "bold": "\033[1m",
    "reset": "\033[0m",
    "planner": "\033[94m",
    "researcher": "\033[92m",
    "writer": "\033[93m",
    "critic": "\033[91m",
    "ok": "\033[92m",
    "warn": "\033[93m",
}
_COLOR = True


def paint(key: str, text: str) -> str:
    if not _COLOR:
        return text
    return f"{C.get(key, '')}{text}{C['reset']}"


def printer(*, quiet: bool = False):
    """把编排事件渲染成终端输出。"""

    def on_event(ev: dict):
        if quiet:
            return
        kind = ev.get("type")
        if kind == "run_start":
            print()
            print(paint("bold", "格物 GeWu · 多 Agent 深度研究"))
            print(paint("dim", f"课题:{ev['topic']}"))
            print(paint("dim", f"参数:{ev['depth']} 个子问题 · {'联网检索' if ev['search'] else '纯模型知识'} · 模型 {ev['model']}"))
            print(paint("dim", "─" * 62))
        elif kind == "task_start":
            task = ev["task"]
            meta = roles.ROLE_META.get(task["role"], {})
            head = f"{meta.get('emoji', '•')} {meta.get('name', task['role'])}"
            print(f"{paint(task['role'], head)}  {paint('dim', task['title'])}")
        elif kind == "task_log":
            print(paint("dim", f"   ├ {ev['text']}"))
        elif kind == "task_end":
            task = ev["task"]
            if task["status"] == "failed":
                print(paint("warn", f"   ✗ {task['title']} — {ev.get('error', '')[:100]}"))
            else:
                reused = "♻️ 复用" if task["status"] == "reused" else "✓"
                print(paint("ok", f"   {reused} {ev.get('summary', '')} · {task.get('elapsed') or 0}s"))
        elif kind == "log":
            print(paint("dim", ev.get("text", "")))
        elif kind == "final":
            print(paint("dim", "─" * 62))
            print(paint("bold", f"✅ 报告完成 · {ev.get('chars', 0)} 字"))
        elif kind == "run_end":
            print(paint("dim", f"   耗时 {ev['elapsed']}s · {ev['tasks']} 条任务 · 产物 {ev['report_path']}"))
        elif kind == "error":
            print(paint("warn", f"✗ {ev.get('error', '')}"))

    return on_event


def _select_llm(args):
    if getattr(args, "mock", False):
        return mock.make_mock_llm(), "mock(离线演示)"
    cfg = config_from_args(args)
    if not cfg.get("api_key"):
        raise LLMError("未配置 LLM_API_KEY:复制 .env.example 为 .env 填入 key,或加 --mock 离线演示")
    return (lambda messages, **kw: chat(messages, config=cfg, **kw)), cfg.get("model") or "unknown"


def cmd_research(args) -> int:
    if not args.topic:
        print("请给出课题,例如:python -m gewu \"大模型会对就业产生什么影响\"")
        return 2
    try:
        llm, label = _select_llm(args)
    except LLMError as e:
        print(paint("warn", f"✗ {e}"))
        return 2

    resume_dir = os.path.join(runs_root_path(), args.resume) if args.resume else None
    if resume_dir and not os.path.isdir(resume_dir):
        print(paint("warn", f"✗ 找不到要续跑的 run:{args.resume}"))
        return 2

    orch = Orchestrator(
        args.topic,
        depth=args.depth,
        use_search=not args.no_search,
        llm=llm,
        on_event=printer(quiet=args.quiet),
        resume_dir=resume_dir,
        search_limit=args.search_limit,
        revise=not args.no_revise,
        model_label=label,
    )
    result = orch.run()

    out = args.out
    if out:
        out_dir = os.path.dirname(os.path.abspath(out))
        os.makedirs(out_dir, exist_ok=True)
        shutil.copyfile(result.report_path, out)
        print(paint("ok", f"→ 已另存为 {out}"))
    if args.json:
        print(json.dumps(
            {
                "run_id": result.run_id,
                "report_path": result.report_path,
                "chars": count_words(result.report),
                "words": count_words(result.report),
                "sources": len(result.sources),
                "elapsed": result.elapsed,
                "tasks": [t.to_dict() for t in result.tasks],
                "critique": result.critique.to_dict() if result.critique else None,
                "plan": result.plan.to_dict() if result.plan else None,
            },
            ensure_ascii=False,
            indent=2,
        ))
    if not args.quiet:
        print()
        print(paint("dim", f"阅读全文:python -m gewu show {result.run_id}"))
    return 0


def cmd_plan(args) -> int:
    if not args.topic:
        print("请给出课题")
        return 2
    try:
        llm, _ = _select_llm(args)
    except LLMError as e:
        print(paint("warn", f"✗ {e}"))
        return 2
    plan_obj, _raw = roles.plan(args.topic, args.depth, llm)
    if args.json:
        print(json.dumps(plan_obj.to_dict(), ensure_ascii=False, indent=2))
        return 0
    print()
    print(paint("bold", f"🧭 拆题结果 · {args.topic}"))
    print(paint("dim", f"切分角度:{plan_obj.angle}"))
    print()
    for s in plan_obj.subquestions:
        print(f"  {paint('planner', s.id)} {s.question}")
        if s.why:
            print(paint("dim", f"      为什么要它:{s.why}"))
        print(paint("dim", f"      查询词:{' / '.join(s.queries)}"))
    return 0


def cmd_runs(args) -> int:
    runs = list_runs()
    if not runs:
        print("还没有任何 run,先跑一次:python -m gewu \"课题\"")
        return 0
    print()
    print(paint("bold", f"历史 run({len(runs)} 次)"))
    print(paint("dim", "─" * 72))
    for r in runs:
        print(f"{r.get('run_id', '?'):<20} {r.get('chars', 0):>5} 字  {r.get('sources', 0):>3} 来源  "
              f"{paint('dim', truncate(r.get('topic', ''), 34))}")
    return 0


def cmd_show(args) -> int:
    run_dir = os.path.join(runs_root_path(), args.run_id)
    if not os.path.isdir(run_dir):
        print(paint("warn", f"✗ 找不到 run:{args.run_id}"))
        return 2
    if args.tasks:
        path = os.path.join(run_dir, "tasks.json")
        if not os.path.isfile(path):
            print("该 run 没有 tasks.json")
            return 1
        with open(path, encoding="utf-8") as f:
            tasks = json.load(f)
        print()
        print(paint("bold", f"任务看板 · {args.run_id}"))
        for t in tasks:
            meta = roles.ROLE_META.get(t["role"], {})
            print(f"  {t['id']:<4} {meta.get('emoji', '')} {meta.get('name', t['role']):<4} "
                  f"{t['status']:<7} {t['elapsed'] or 0:>5}s  {truncate(t['title'], 40)}")
        return 0
    path = os.path.join(run_dir, "report.md")
    if not os.path.isfile(path):
        print("该 run 还没有 report.md")
        return 1
    with open(path, encoding="utf-8") as f:
        print(f.read())
    return 0


def cmd_web(args) -> int:
    from .web import serve

    serve(host=args.host, port=args.port, mock_mode=args.mock)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gewu",
        description="格物 GeWu · 多 Agent 深度研究助手(规划者 / 研究者 / 写作者 / 审稿者)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "例子:\n"
            "  python -m gewu \"大模型会对就业产生什么影响\"\n"
            "  python -m gewu \"县域充电桩为什么建得慢\" --depth 5 --out data/report.md\n"
            "  python -m gewu --mock \"AI 会不会取代产品经理\"   # 离线演示\n"
            "  python -m gewu web --port 8000                    # 网页看编排过程\n"
        ),
    )
    parser.add_argument("--version", action="version", version=f"gewu {__version__}")
    sub = parser.add_subparsers(dest="cmd")

    common = dict()
    r = sub.add_parser("research", help="跑一次完整的深度研究(默认命令)")
    r.add_argument("topic", nargs="?", help="研究课题")
    r.add_argument("--depth", type=int, default=4, help="拆成几个子问题(1~6,默认 4)")
    r.add_argument("--no-search", action="store_true", help="不联网,只用模型知识")
    r.add_argument("--no-revise", action="store_true", help="跳过审稿后的修订轮")
    r.add_argument("--search-limit", type=int, default=4, help="每个查询词取几条结果(默认 4)")
    r.add_argument("--resume", help="复用指定 run 里已完成的任务,继续往下跑")
    r.add_argument("--out", help="把终稿另存到指定路径")
    r.add_argument("--json", action="store_true", help="输出结构化 JSON")
    r.add_argument("--mock", action="store_true", help="离线演示模式(不调模型、不联网)")
    r.add_argument("--quiet", action="store_true", help="只输出结果,不打印过程")
    r.add_argument("--api-key")
    r.add_argument("--base-url")
    r.add_argument("--model")

    p = sub.add_parser("plan", help="只跑规划者,看它怎么拆题")
    p.add_argument("topic", nargs="?")
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--json", action="store_true")
    p.add_argument("--mock", action="store_true")
    p.add_argument("--api-key")
    p.add_argument("--base-url")
    p.add_argument("--model")

    l = sub.add_parser("runs", help="列出历史 run")

    s = sub.add_parser("show", help="打印某次 run 的终稿或任务看板")
    s.add_argument("run_id")
    s.add_argument("--tasks", action="store_true", help="打印任务看板而不是终稿")

    w = sub.add_parser("web", help="启动网页界面(可视化四个角色的编排过程)")
    w.add_argument("--host", default="127.0.0.1")
    w.add_argument("--port", type=int, default=8000)
    w.add_argument("--mock", action="store_true", help="网页也跑离线演示模式")

    return parser


KNOWN = {"research", "plan", "runs", "show", "web"}


def main(argv: list[str] | None = None) -> int:
    global _COLOR
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    _COLOR = sys.stdout.isatty() and os.environ.get("NO_COLOR") is None

    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        build_parser().print_help()
        return 0
    if not any(tok in KNOWN for tok in argv) and not any(tok in ("-h", "--help", "--version") for tok in argv):
        argv.insert(0, "research")

    args = build_parser().parse_args(argv)
    if args.cmd == "plan":
        return cmd_plan(args)
    if args.cmd == "runs":
        return cmd_runs(args)
    if args.cmd == "show":
        return cmd_show(args)
    if args.cmd == "web":
        return cmd_web(args)
    return cmd_research(args)
