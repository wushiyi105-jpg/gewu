"""离线单元测试:不消耗 token、不依赖网络。

运行:python -m unittest discover tests -v
"""

import contextlib
import io
import json
import os
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gewu import cli, llm, mock, roles, search, utils  # noqa: E402
from gewu.orchestrator import Orchestrator, list_runs  # noqa: E402

FIXTURE = """
<html><body>
<div class="result results_links">
  <a rel="nofollow" class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fa&amp;rut=1">
    大模型 <b>就业</b> 影响研究
  </a>
  <a class="result__snippet" href="https://example.com/a">这是一段 <b>摘要</b> 文本,讲替代效应。</a>
</div>
<div class="result results_links">
  <a rel="nofollow" class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.org%2Fb&amp;rut=2">第二条 结果</a>
  <a class="result__snippet" href="https://example.org/b">第二条摘要</a>
</div>
<div class="result result--ad">
  <a class="result__a" href="//duckduckgo.com/y.js?ad_provider=x">广告不该出现</a>
</div>
</body></html>
"""


BING_FIXTURE = """
<ul>
<li class="b_algo" data-id="1"><h2><a target="_blank" href="https://example.com/a">大模型 <strong>就业</strong> 研究</a></h2>
  <div class="b_caption"><p class="b_lineclamp2">第一条摘要…</p></div></li>
<li class="b_algo" data-id="2"><h2><a href="https://www.bing.com/ck/a?!&&p=1&u=a1aHR0cHM6Ly9leGFtcGxlLm9yZy9i&ntb=1">第二条结果</a></h2>
  <p class="b_lineclamp3">第二条摘要</p></li>
<li class="b_algo" data-id="3"><h2><a href="https://www.bing.com/images/search?q=x">站内导航</a></h2></li>
</ul>
"""


SOGOU_FIXTURE = """
<div class="vrwrap"><h3 class="vr-title"><a target="_blank" href="http://www.chinareports.org.cn/index/news/88967.html">
   AI重塑<em>就业</em>市场</a></h3>
  <div class="base-ellipsis clamp2 card_normal_result__summary_1a50">以<em>大模型</em>为代表的技术正在重塑就业市场,预测到2030年…</div></div>
<div class="vrwrap"><h3 class="vr-title"><a href="/link?url=abc">跳转型结果</a></h3></div>
"""


SO360_FIXTURE = """
<ul class="result">
<li class="res-list" data-result-type="normal" data-result-pos="2">
  <h3   class="res-title"><a href="https://www.so.com/link?m=abc" data-mdurl="https://example.com/a" rel="noopener" target="_blank"><em>大模型</em>技术发展与<em>就业</em>影响分析</a></h3>
  <p class="res-desc"><span class="gray">2026年3月21日</span><span data-result-summary>一段摘要,讲替代效应</span></p>
</li>
<li class="res-list"><div class="g-mohe">AI 卡片,没有真实外链</div></li>
<li class="res-list" data-result-type="normal">
  <h3 class="res-title " ><a href="https://www.so.com/link?m=def" data-mdurl="https://example.org/b">第二条结果</a></h3>
</li>
</ul>
"""


def fake_search(results=None, calls=None, error=None):
    """返回一个假的 search_many,行为可控。"""

    def _fn(queries, limit=4, **kwargs):
        if calls is not None:
            calls.append(list(queries))
        if error:
            raise error
        items = (results or [])[: limit or 4]
        return [search.SearchResult(t, u, s) for t, u, s in items], []

    return _fn


class TestLLM(unittest.TestCase):
    def test_extract_json_plain_and_fenced(self):
        self.assertEqual(llm.extract_json('{"a": 1}'), {"a": 1})
        self.assertEqual(llm.extract_json('```json\n{"a": 2}\n```'), {"a": 2})
        self.assertEqual(llm.extract_json('说明文字 {"a": 3} 后缀'), {"a": 3})

    def test_extract_json_tolerates_nesting(self):
        data = llm.extract_json('前言 {"a": {"b": [1, 2]}} 后记')
        self.assertEqual(data["a"]["b"], [1, 2])

    def test_extract_json_raises_on_garbage(self):
        with self.assertRaises(llm.LLMError):
            llm.extract_json("完全没有 JSON")

    def test_config_from_args_overrides(self):
        class Args:
            api_key = "k-test"
            base_url = "http://localhost:11434/v1"
            model = "qwen2.5"

        cfg = llm.config_from_args(Args())
        self.assertEqual(cfg["api_key"], "k-test")
        self.assertEqual(cfg["base_url"], "http://localhost:11434/v1")
        self.assertEqual(cfg["model"], "qwen2.5")


class TestSearch(unittest.TestCase):
    def test_parse_results_and_decode_redirect(self):
        results = search.parse_html(FIXTURE)
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].url, "https://example.com/a")
        self.assertIn("大模型", results[0].title)
        self.assertIn("摘要", results[0].snippet)
        self.assertEqual(results[1].url, "https://example.org/b")

    def test_ad_link_is_dropped(self):
        urls = [r.url for r in search.parse_html(FIXTURE)]
        self.assertNotIn("", urls)
        self.assertTrue(all(u.startswith("https://") for u in urls))

    def test_clean_url_passthrough(self):
        self.assertEqual(search._clean_url("https://example.com/x"), "https://example.com/x")
        self.assertEqual(search._clean_url("//example.com/x"), "https://example.com/x")
        self.assertEqual(search._clean_url(""), "")

    def test_empty_query_returns_empty_without_network(self):
        self.assertEqual(search.search("   "), [])

    def test_parse_bing_blocks_and_decode_ck_link(self):
        results = search.parse_bing(BING_FIXTURE)
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].title, "大模型 就业 研究")
        self.assertEqual(results[0].provider, "bing")
        self.assertEqual(results[1].url, "https://example.org/b")

    def test_parse_sogou(self):
        results = search.parse_sogou(SOGOU_FIXTURE)
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].url, "http://www.chinareports.org.cn/index/news/88967.html")
        self.assertIn("就业", results[0].title)
        self.assertIn("2030", results[0].snippet)
        self.assertTrue(results[1].url.startswith("https://www.sogou.com/link?"))

    def test_irrelevant_results_are_dropped(self):
        junk = [search.SearchResult("大（汉语文字）_百度百科", "https://baike.baidu.com/item/大", "读音")]
        self.assertEqual(search._relevant(junk, "大模型 就业"), [])
        good = [search.SearchResult("大模型 就业 影响", "https://example.com/x", "报告")]
        self.assertEqual(len(search._relevant(good, "大模型 就业")), 1)

    def test_parse_360_uses_mdurl_and_skips_ai_card(self):
        results = search.parse_360(SO360_FIXTURE)
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].url, "https://example.com/a")
        self.assertEqual(results[0].title, "大模型技术发展与就业影响分析")
        self.assertIn("替代效应", results[0].snippet)
        self.assertEqual(results[1].url, "https://example.org/b")

    def test_long_question_query_still_matches(self):
        # 规划者偶尔把整句问句当查询词,二元组重合度要能救回来
        hits = [search.SearchResult("大模型正在重塑就业结构", "https://example.com/a", "生成式 AI 对就业的影响")]
        self.assertEqual(len(search._relevant(hits, "大模型会对就业产生什么影响")), 1)

    def test_provider_fallback(self):
        old = dict(search.PROVIDERS)
        old_pref = search._preferred
        search.PROVIDERS["broken"] = lambda q, l, t: (_ for _ in ()).throw(search.SearchError("被掐断"))
        search.PROVIDERS["okay"] = lambda q, l, t: [search.SearchResult("任意查询 结果", "https://example.com/x", "摘要")]
        try:
            results = search.search("任意查询", providers=["broken", "okay"])
            self.assertEqual(len(results), 1)
            self.assertEqual(results[0].url, "https://example.com/x")
            with self.assertRaises(search.SearchError):
                search.search("任意查询", providers=["broken"])
        finally:
            search.PROVIDERS = old
            search._preferred = old_pref


class TestRoles(unittest.TestCase):
    def test_normalize_plan_from_dict(self):
        raw = {
            "angle": "事实→机制",
            "subquestions": [
                {"question": "问题一?", "why": "因为", "queries": ["q1", "q2"]},
                {"question": "问题二?", "queries": ["q3"]},
                {"question": "问题三?"},
                {"question": "问题四?"},
            ],
        }
        plan = roles.normalize_plan(raw, depth=3)
        self.assertEqual([s.id for s in plan.subquestions], ["Q1", "Q2", "Q3"])
        self.assertEqual(plan.angle, "事实→机制")
        self.assertEqual(plan.subquestions[0].queries, ["q1", "q2"])
        # 没给 query 时用问题本身兜底
        self.assertIn("问题三", plan.subquestions[2].queries[0])

    def test_normalize_plan_from_strings(self):
        plan = roles.normalize_plan(["甲问题", "乙问题"], depth=2)
        self.assertEqual(len(plan.subquestions), 2)
        self.assertEqual(plan.subquestions[1].question, "乙问题")

    def test_normalize_plan_empty_falls_back(self):
        plan = roles.normalize_plan({"subquestions": []}, depth=2, topic="测试课题")
        self.assertEqual(len(plan.subquestions), 2)
        self.assertIn("测试课题", plan.subquestions[0].question)

    def test_planner_retries_when_json_unparseable(self):
        calls = []

        def flaky(messages, **kwargs):
            calls.append(messages)
            if len(calls) == 1:
                return "我需要先想一想,这个课题很大。"
            return json.dumps(
                {"angle": "事实→机制", "subquestions": [{"id": "Q1", "question": "q?", "queries": ["k1"]}]},
                ensure_ascii=False,
            )

        plan, _ = roles.plan("课题", 2, flaky)
        self.assertEqual(len(calls), 2)
        self.assertEqual(plan.angle, "事实→机制")

    def test_fallback_plan_queries_carry_facet(self):
        plan = roles.fallback_plan("大模型 就业", 4)
        self.assertIn("现状", plan.subquestions[0].queries[0])
        self.assertNotEqual(plan.subquestions[0].queries[0], plan.subquestions[1].queries[0])

    def test_ensure_queries_drops_sentence_like_query(self):
        qs = roles.ensure_queries("大模型 就业", "大模型会对就业产生什么影响?", ["大模型会对就业产生什么影响?"])
        self.assertEqual(len(qs), 1)
        self.assertNotEqual(qs[0], "大模型会对就业产生什么影响?")

    def test_normalize_note_refs_and_strings(self):
        note = roles.normalize_note(
            {"findings": [{"point": "结论", "detail": "细节", "sources": ["[S3]", "S 5"]}, "一条纯文本"],
             "gaps": ["缺数据"]},
            "原问题",
        )
        self.assertEqual(note.findings[0].sources, ["S3", "S5"])
        self.assertEqual(note.findings[0].confidence, "sourced")
        self.assertEqual(note.findings[1].confidence, "model_knowledge")
        self.assertEqual(note.gaps, ["缺数据"])

    def test_normalize_critique_strings(self):
        c = roles.normalize_critique({"verdict": "revise", "issues": ["引用不足", {"severity": "high", "problem": "越界"}]})
        self.assertEqual(len(c.issues), 2)
        self.assertEqual(c.issues[1]["severity"], "high")
        self.assertEqual(len(c.high_issues), 1)

    def test_mock_roles_produce_structured_output(self):
        llm_fn = mock.make_mock_llm()
        plan, raw = roles.plan("大模型会对就业产生什么影响", 4, llm_fn)
        self.assertEqual(len(plan.subquestions), 4)
        notes = [roles.research(s.question, "【检索材料】\n[S1] 某来源\n    URL: https://example.com", llm_fn)[0]
                 for s in plan.subquestions]
        self.assertTrue(all(n.findings for n in notes))
        draft, _ = roles.write_report("大模型会对就业产生什么影响", plan, notes, llm_fn)
        self.assertIn("## 摘要", draft)
        self.assertIn("## 结论", draft)
        critique, _ = roles.critique_report("大模型会对就业产生什么影响", draft, notes, llm_fn)
        self.assertEqual(critique.verdict, "revise")
        final, _ = roles.revise_report("大模型会对就业产生什么影响", draft, critique, llm_fn)
        self.assertIn("## 结论", final)
        self.assertGreater(len(final), len(draft))


class TestUtils(unittest.TestCase):
    def test_strip_fences(self):
        self.assertEqual(utils.strip_md_fences("```markdown\n# 标题\n```"), "# 标题")

    def test_count_words(self):
        self.assertEqual(utils.count_words("中文字abc"), 3 + 1)

    def test_truncate(self):
        self.assertTrue(utils.truncate("很长的文本" * 10, 10).endswith("…"))


class BaseRun(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.llm = mock.make_mock_llm()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def run_orch(self, **kwargs):
        params = dict(depth=3, use_search=False, llm=self.llm, runs_root=self.tmp, model_label="mock")
        params.update(kwargs)
        return Orchestrator(params.pop("topic", "大模型对就业的影响"), **params).run()


class TestOrchestrator(BaseRun):
    def test_full_chain_without_search(self):
        result = self.run_orch()
        self.assertTrue(result.ok)
        self.assertEqual(len(result.tasks), 7)  # 1 规划 + 3 研究 + 初稿 + 审稿 + 修订(depth=3)
        self.assertTrue(all(t.status == "done" for t in result.tasks))
        for section in ("## 摘要", "## 结论", "## 参考资料", "## 附录 · 协作任务看板"):
            self.assertIn(section, result.report)
        self.assertIn("未做外部核验", result.report)
        self.assertEqual(len(result.sources), 0)
        for name in ("report.md", "tasks.json", "cache.json", "meta.json"):
            self.assertTrue(os.path.isfile(os.path.join(result.run_dir, name)), name)
        self.assertTrue(os.path.isfile(os.path.join(result.run_dir, "tasks", "T1_planner.json")))

    def test_search_adds_real_citations(self):
        calls = []
        fake = fake_search(
            results=[("来源甲", "https://example.com/a", "摘要甲"), ("来源乙", "https://example.org/b", "摘要乙")],
            calls=calls,
        )
        result = self.run_orch(use_search=True, search_fn=fake)
        self.assertGreaterEqual(len(result.sources), 2)
        self.assertEqual(result.sources[0].sid, "S1")
        self.assertIn("example.com", result.report)
        self.assertIn("[S", result.report)
        self.assertGreaterEqual(len(calls), 3)

    def test_search_failure_degrades_instead_of_crashing(self):
        result = self.run_orch(use_search=True, search_fn=fake_search(error=search.SearchError("被限流")))
        self.assertTrue(result.ok)
        self.assertEqual(len(result.sources), 0)
        self.assertIn("## 摘要", result.report)

    def test_sources_are_registered_once(self):
        fake = fake_search(
            results=[("来源甲", "https://example.com/a", "摘要甲"), ("来源乙", "https://example.org/b", "摘要乙")]
        )
        result = self.run_orch(use_search=True, search_fn=fake)
        sids = [s.sid for s in result.sources]
        self.assertEqual(len(sids), len(set(sids)))  # 每条来源只登记一次
        self.assertEqual(result.report.count("## 参考资料"), 1)
        self.assertEqual(sids, [f"S{i + 1}" for i in range(len(sids))])

    def test_ensure_sections_fills_missing_parts(self):
        orch = Orchestrator("课题", depth=2, use_search=False, llm=self.llm, runs_root=self.tmp)
        notes = [roles.Note(question="q", findings=[], gaps=["缺 2024 年数据"], takeaway="判断一")]
        body = orch._ensure_sections("## 01 / 一节\n\n正文。", notes)
        self.assertIn("## 摘要", body)
        self.assertIn("缺 2024 年数据", body)
        self.assertIn("## 结论", body)
        self.assertIn("判断一", body)

    def test_short_revision_is_rejected(self):
        base = self.llm

        def thin_revision(messages, **kwargs):
            system = "".join(m["content"] for m in messages if m.get("role") == "system")
            if "修订" in system:
                return "## 摘要\n\n改短了。"
            return base(messages, **kwargs)

        result = self.run_orch(llm=thin_revision)
        self.assertTrue(result.ok)
        self.assertIn("## 结论", result.report)  # 保留初稿而不是残缺修订稿
        self.assertTrue(any("保留初稿" in t.summary for t in result.tasks))

    def test_writer_failure_falls_back_to_notes(self):
        base = self.llm

        def flaky(messages, **kwargs):
            system = "".join(m["content"] for m in messages if m.get("role") == "system")
            if "写作者" in system and "修订" not in system:
                raise llm.LLMError("写作模型挂了")
            return base(messages, **kwargs)

        result = self.run_orch(llm=flaky)
        self.assertTrue(result.ok)
        self.assertIn("笔记直出", result.report)
        self.assertTrue(any(t.status == "failed" for t in result.tasks))

    def test_resume_reuses_finished_tasks(self):
        first = self.run_orch()
        calls = []

        def counting(messages, **kwargs):
            calls.append(1)
            return self.llm(messages, **kwargs)

        second = Orchestrator(
            "大模型对就业的影响", depth=3, use_search=False, llm=counting, runs_root=self.tmp,
            resume_dir=first.run_dir, model_label="mock",
        ).run()
        self.assertEqual(calls, [])  # 所有任务的输入没变,整条链路都命中缓存
        self.assertTrue(any(t.status == "reused" for t in second.tasks))
        self.assertIn("## 结论", second.report)

    def test_list_runs_reads_meta(self):
        self.run_orch(topic="列表测试")
        runs = list_runs(self.tmp)
        self.assertEqual(len(runs), 1)
        self.assertEqual(runs[0]["topic"], "列表测试")
        self.assertGreater(runs[0]["chars"], 0)


class TestCLI(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.old = os.environ.get("GEWU_RUNS_ROOT")
        os.environ["GEWU_RUNS_ROOT"] = self.tmp

    def tearDown(self):
        if self.old is None:
            os.environ.pop("GEWU_RUNS_ROOT", None)
        else:
            os.environ["GEWU_RUNS_ROOT"] = self.old
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_mock_run_via_cli(self):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            code = cli.main(["--mock", "--no-search", "--quiet", "命令行测试课题"])
        self.assertEqual(code, 0)
        runs = list_runs(self.tmp)
        self.assertEqual(len(runs), 1)
        self.assertGreater(runs[0]["chars"], 100)
        self.assertTrue(os.path.isfile(os.path.join(runs[0]["dir"], "report.md")))

    def test_plan_subcommand(self):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            code = cli.main(["plan", "--mock", "拆题测试", "--json"])
        self.assertEqual(code, 0)
        data = json.loads(buf.getvalue())
        self.assertEqual(len(data["subquestions"]), 4)

    def test_web_page_is_renderable(self):
        from gewu.web import HTML

        for token in ("格物 GeWu", "规划者", "研究者", "审稿者", "/api/run"):
            self.assertIn(token, HTML)


if __name__ == "__main__":
    unittest.main(verbosity=2)
