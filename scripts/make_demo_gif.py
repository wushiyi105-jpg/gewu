"""生成 README 用的终端演示动图(开发脚本,需要 Pillow)。

    python scripts/make_demo_gif.py

输出 assets/demo.gif:960x600,深色底,模拟一次真实编排的终端输出。
"""

from __future__ import annotations

import os
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

W, H = 960, 600
BG = (11, 15, 20)
BAR = (22, 28, 36)
FG = (200, 208, 218)
DIM = (120, 132, 148)
ACCENT = {"planner": (122, 162, 247), "researcher": (158, 206, 106),
          "writer": (224, 175, 104), "critic": (247, 118, 142), "ok": (115, 218, 202)}

FONT_PATHS = [
    r"C:\Windows\Fonts\msyh.ttc",
    r"C:\Windows\Fonts\consola.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
]
SIZE, LH, LEFT, TOP = 15, 25, 26, 62


def load_font(size=SIZE) -> ImageFont.FreeTypeFont:
    for p in FONT_PATHS:
        if os.path.isfile(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


FONT = load_font()


def frame(lines: list[tuple[str, tuple[int, int, int]]], cursor: bool = True) -> Image.Image:
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, W, 34], fill=BAR)
    for i, c in enumerate([(255, 95, 86), (255, 189, 46), (39, 201, 63)]):
        d.ellipse([14 + i * 20, 12, 24 + i * 20, 22], fill=c)
    d.text((84, 9), "gewu — python -m gewu", font=FONT, fill=DIM)

    y = TOP
    for text, color in lines:
        if y > H - 18:
            break
        d.text((LEFT, y), text, font=FONT, fill=color)
        y += LH

    if cursor:
        d.rectangle([LEFT, y + 3, LEFT + 9, y + 17], fill=(200, 208, 218))
    return img


def build_frames() -> list[tuple[Image.Image, int]]:
    rows: list[tuple[str, tuple[int, int, int]]] = []

    def add(*items: tuple[str, tuple[int, int, int]], hold: int = 320):
        rows.extend(items)
        frames.append((frame(rows), hold))

    frames: list[tuple[Image, int]] = []

    # 1) 敲命令
    cmd = '$ python -m gewu "大模型会对就业产生什么影响"'
    for i in range(6, len(cmd) + 1, 12):
        frames.append((frame([(cmd[:i], FG)]), 140))
    add((cmd, FG), hold=700)

    # 2) 表头
    add(("格物 GeWu · 多 Agent 深度研究", FG), hold=500)
    add(("课题:大模型会对就业产生什么影响", DIM),
        ("参数:4 个子问题 · 联网检索 · 模型 deepseek-flash", DIM),
        ("─" * 62, (60, 68, 80)), hold=700)

    # 3) 规划者
    add(("🧭 规划者  拆解课题", ACCENT["planner"]),
        ("   ├ Q1 已在哪些任务和职业上产生可测量的替代与增效?", FG),
        ("   ├ Q2 冲击在人群、行业、技能层级上如何分布?", FG),
        ("   ├ Q3 替代与生产率提升如何传导到岗位与工资?", FG),
        ("   ├ Q4 学界对「失业」还是「结构转型」分歧何在?", FG),
        ("   ✓ 4 个子问题 · 5.8s", ACCENT["ok"]), hold=1100)

    # 4) 研究者 ×2
    add(("🔍 研究者  Q1 · 已有实证与可测量变化", ACCENT["researcher"]),
        ("   ├ 检索:大模型 岗位替代 实证研究", DIM),
        ("   ├ 命中:人工智能对就业的新一轮冲击调查(新华网)", FG),
        ("   ├ 命中:AI 替代就业,国家出手了(中新网)", FG), hold=900)
    add(("   ✓ 4 条发现 · 3 条有 [S#] 来源 · 41.2s", ACCENT["ok"]), hold=600)
    add(("🔍 研究者  Q2 · 冲击的分布", ACCENT["researcher"]),
        ("   ├ 检索:生成式AI 就业 工资 技能 分化", DIM),
        ("   ✓ 3 条发现 · 2 条有来源 · 33.7s", ACCENT["ok"]), hold=900)
    add(("🔍 研究者  Q3 · 传导机制", ACCENT["researcher"]),
        ("🔍 研究者  Q4 · 争议与政策", ACCENT["researcher"]),
        ("   ✓ 4 个研究任务完成 · 累计 8 条来源", ACCENT["ok"]), hold=900)

    # 5) 写作者 + 审稿者
    add(("✍️ 写作者  整合成文 · 初稿", ACCENT["writer"]),
        ("   ✓ 1832 字 · 4 节 · 引用 8 条 · 12.4s", ACCENT["ok"]), hold=800)
    add(("🧐 审稿者  挑刺:论据 / 逻辑 / 遗漏", ACCENT["critic"]),
        ("   ├ ● high  结论超出证据:把相关性写成了因果", (247, 118, 142)),
        ("   ├ ● high  「30% 岗位消失」无 [S#] 支撑", (247, 118, 142)),
        ("   ├ ● medium 缺少反面证据与边界条件", (224, 175, 104)),
        ("   ✓ verdict=revise · 4 条意见(2 高) · 9.6s", ACCENT["ok"]), hold=1200)

    # 6) 修订 + 收尾
    add(("✍️ 写作者  按审稿意见修订", ACCENT["writer"]),
        ("   ✓ 终稿 2375 字 · 补 3 处引用、降级 2 处口述 · 47.3s", ACCENT["ok"]), hold=900)
    add(("─" * 62, (60, 68, 80)),
        ("✅ 完成 · 8 个任务 · 172.7s · 来源 23 条", ACCENT["ok"]),
        ("📄 runs/20261124-212734/report.md", FG), hold=1600)
    add(("$ python -m gewu show 20261124-212734 --tasks", DIM), hold=1300)
    return frames


def main() -> None:
    out = Path(__file__).resolve().parent.parent / "assets" / "demo.gif"
    out.parent.mkdir(parents=True, exist_ok=True)
    frames = build_frames()
    imgs = [f for f, _ in frames]
    imgs[0].save(out, save_all=True, append_images=imgs[1:],
                 duration=[d for _, d in frames], loop=0, optimize=True)
    print(f"已生成 {out} · {len(frames)} 帧 · {out.stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    main()
