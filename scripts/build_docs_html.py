#!/usr/bin/env python3
"""把 docs/*.md 按分类渲染进总览 HTML（可重复执行，标记区间替换）。"""
from __future__ import annotations

import html
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
PAGE = DOCS / "修复与亮点总览-2026-11-26.html"

CATEGORIES = [
    ("总览与优化", "lib-opt", [
        "修复与亮点总览-2026-11-26.md",
        "优化-n-gram与FP8-KV-2026-11-24.md",
    ]),
    ("验收与验证", "lib-verify", [
        "验证最终结果-2026-11-23.md",
        "HTTP压测与最终验收-2026-11-24.md",
        "验证包结果-2026-11-19.md",
        "验收结果-2026-11-18.md",
        "集群验收进度-2026-11-17.md",
    ]),
    ("审计与评审", "lib-audit", [
        "全项目QA审计与修复-2026-11-18.md",
        "cascade-infer-问题与修复报告.md",
        "flash代码复核与修复-2026-11-17.md",
    ]),
    ("规划与路线", "lib-plan", [
        "上线路线图.md",
    ]),
]


def inline(text: str) -> str:
    s = html.escape(text)
    s = re.sub(r"`([^`]+)`", r"<code>\1</code>", s)
    s = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", s)
    s = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<a href="\2" target="_blank" rel="noreferrer">\1</a>', s)
    return s


def is_special(line: str) -> bool:
    return bool(
        line.startswith("#")
        or line.startswith("```")
        or line.startswith(">")
        or line.strip().startswith("|")
        or re.match(r"^\s*([-*+]|\d+\.)\s+", line)
        or re.match(r"^\s*---+\s*$", line)
    )


def parse_list(lines: list[str], i: int) -> tuple[str, int]:
    m0 = re.match(r"^(\s*)([-*+]|\d+\.)\s+", lines[i])
    assert m0
    base = len(m0.group(1))
    ordered = bool(re.match(r"^\s*\d+\.", lines[i]))
    tag = "ol" if ordered else "ul"
    items: list[str] = []
    while i < len(lines):
        m = re.match(r"^(\s*)([-*+]|\d+\.)\s+(.*)$", lines[i])
        if not m or len(m.group(1)) != base:
            break
        content = m.group(3).strip()
        i += 1
        # 续行（缩进的普通文本）并入当前条目
        while (
            i < len(lines)
            and lines[i].strip()
            and not re.match(r"^\s*([-*+]|\d+\.)\s+", lines[i])
            and not lines[i].startswith("#")
            and not lines[i].startswith("```")
            and not lines[i].strip().startswith("|")
        ):
            content += " " + lines[i].strip()
            i += 1
        sub = ""
        if i < len(lines):
            m2 = re.match(r"^(\s*)([-*+]|\d+\.)\s+", lines[i])
            if m2 and len(m2.group(1)) > base:
                sub, i = parse_list(lines, i)
        items.append(f"<li>{inline(content)}{sub}</li>")
    return f"<{tag}>" + "".join(items) + f"</{tag}>", i


def convert(md: str) -> str:
    lines = md.splitlines()
    out: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if not line.strip():
            i += 1
            continue
        if line.startswith("```"):
            i += 1
            code: list[str] = []
            while i < len(lines) and not lines[i].startswith("```"):
                code.append(lines[i])
                i += 1
            i += 1
            out.append("<pre><code>" + html.escape("\n".join(code)) + "</code></pre>")
            continue
        m = re.match(r"^(#{1,6})\s+(.*)$", line)
        if m:
            level = min(6, len(m.group(1)) + 3)
            out.append(f"<h{level}>{inline(m.group(2))}</h{level}>")
            i += 1
            continue
        if line.strip().startswith("|") and i + 1 < len(lines) and re.match(
            r"^\s*\|?[\s:|-]+\|", lines[i + 1]
        ):
            rows: list[list[str]] = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                if not re.match(r"^[\s:|-]+$", "".join(cells)):
                    rows.append(cells)
                i += 1
            if rows:
                head = "".join(f"<th>{inline(c)}</th>" for c in rows[0])
                body = "".join(
                    "<tr>" + "".join(f"<td>{inline(c)}</td>" for c in r) + "</tr>"
                    for r in rows[1:]
                )
                out.append(f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>")
            continue
        if re.match(r"^\s*([-*+]|\d+\.)\s+", line):
            block, i = parse_list(lines, i)
            out.append(block)
            continue
        if line.startswith(">"):
            quote: list[str] = []
            while i < len(lines) and lines[i].startswith(">"):
                quote.append(lines[i].lstrip("> ").rstrip())
                i += 1
            out.append("<blockquote>" + inline(" ".join(quote)) + "</blockquote>")
            continue
        if re.match(r"^\s*---+\s*$", line):
            out.append("<hr />")
            i += 1
            continue
        para = [line.strip()]
        i += 1
        while i < len(lines) and lines[i].strip() and not is_special(lines[i]):
            para.append(lines[i].strip())
            i += 1
        out.append("<p>" + inline(" ".join(para)) + "</p>")
    return "\n".join(out)


def doc_block(path: Path) -> str:
    md = path.read_text(encoding="utf-8")
    title = md.splitlines()[0].lstrip("# ").strip() if md else path.stem
    body = convert(md)
    size = path.stat().st_size / 1024
    anchor = "doc-" + re.sub(r"[^0-9a-zA-Z\u4e00-\u9fff]+", "-", path.stem)
    return f"""<details class="doc" id="{anchor}">
  <summary><span class="doc-title">{html.escape(title)}</span><span class="doc-meta">{size:.1f} KB · Markdown 原文</span></summary>
  <div class="doc-body">
{body}
  </div>
</details>"""


def main() -> None:
    page = PAGE.read_text(encoding="utf-8")
    groups: list[str] = []
    navs: list[str] = []
    for name, anchor, files in CATEGORIES:
        blocks = []
        for f in files:
            p = DOCS / f
            if p.exists():
                blocks.append(doc_block(p))
        groups.append(
            f'<div class="doc-group" id="{anchor}"><h3>{name} '
            f'<span class="doc-count">{len(blocks)} 篇</span></h3>\n' + "\n".join(blocks) + "\n</div>"
        )
        navs.append(f'<a href="#{anchor}">{name}</a>')

    library = (
        '<!-- DOC_LIBRARY_START -->\n'
        '<section id="library">\n'
        '  <h2><span class="num">05</span>全部 Markdown 原文（按分类）</h2>\n'
        '  <p>以下内容由 <code>scripts/build_docs_html.py</code> 从 <code>docs/*.md</code> 自动渲染，与源文件一致；点击标题展开，'
        '便于在这个页面里直接查阅全部报告。</p>\n'
        + "\n".join(groups)
        + "\n</section>\n<!-- DOC_LIBRARY_END -->"
    )
    nav = (
        '<!-- DOC_NAV_START -->\n'
        '    <a class="group" href="#library">五、全部 Markdown 原文</a>\n'
        + "\n".join("    " + n for n in navs)
        + "\n    <!-- DOC_NAV_END -->"
    )

    if "<!-- DOC_LIBRARY_START -->" in page:
        page = re.sub(
            r"<!-- DOC_LIBRARY_START -->.*?<!-- DOC_LIBRARY_END -->",
            lambda _m: library,
            page,
            flags=re.S,
        )
    else:
        page = page.replace("<footer>", library + "\n\n<footer>", 1)

    if "<!-- DOC_NAV_START -->" in page:
        page = re.sub(
            r"<!-- DOC_NAV_START -->.*?<!-- DOC_NAV_END -->",
            lambda _m: nav,
            page,
            flags=re.S,
        )
    else:
        page = page.replace(
            '<a class="group" href="#numbers">四、亮点数字</a>',
            '<a class="group" href="#numbers">四、亮点数字</a>\n' + nav,
            1,
        )

    css = """
  /* ---------------- 文档库 ---------------- */
  .doc-group{margin:26px 0}
  .doc-group>h3{font-size:17px;margin:0 0 10px;display:flex;align-items:baseline;gap:10px}
  .doc-count{font-size:11.5px;color:#7f8db0;font-weight:500;letter-spacing:.06em}
  details.doc{border:1px solid var(--line);border-radius:12px;background:rgba(255,255,255,.025);margin:10px 0;overflow:hidden}
  details.doc>summary{cursor:pointer;padding:13px 16px;display:flex;align-items:center;justify-content:space-between;gap:14px;list-style:none}
  details.doc>summary::-webkit-details-marker{display:none}
  details.doc>summary::before{content:"+";display:inline-grid;place-items:center;width:18px;height:18px;border-radius:6px;border:1px solid var(--line);color:#9fb0d0;font-size:12px;flex:none}
  details.doc[open]>summary::before{content:"−"}
  details.doc>summary:hover{background:rgba(129,140,248,.08)}
  .doc-title{font-weight:700;font-size:14px;flex:1}
  .doc-meta{color:#7f8db0;font-size:11.5px;white-space:nowrap}
  .doc-body{padding:4px 20px 20px;border-top:1px solid var(--line)}
  .doc-body h4{margin:16px 0 6px;font-size:15px}
  .doc-body h5{margin:14px 0 5px;font-size:14px;color:#d6deef}
  .doc-body h6{margin:12px 0 4px;font-size:13px;color:#aeb9d0}
  .doc-body p,.doc-body li{font-size:13.5px}
  .doc-body table{font-size:12.5px}
  .doc-body pre{font-size:12px}
"""
    if "details.doc{" not in page:
        page = page.replace("  @media (max-width:860px){", css + "  @media (max-width:860px){", 1)

    PAGE.write_text(page, encoding="utf-8")
    print(f"doc library updated: {sum(len(f) for _, _, f in CATEGORIES)} docs, {len(page)/1024:.1f} KB")


if __name__ == "__main__":
    main()