"""Build results/analysis/report.html from WRITEUP.md: a self-contained report page (figures referenced as figures/*.png).
Needs the `markdown` package (pip install markdown); run: python scripts/build_report.py"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parents[1] / "results" / "analysis"
SRC, OUT = ROOT / "WRITEUP.md", ROOT / "report.html"

md = SRC.read_text()
# drop the H1 (the page has its own masthead) and the italic byline paragraph
md = re.sub(r"^# .*\n", "", md, count=1)
byline = re.search(r"^\*Tim Farrelly.*?\*\s*$", md, flags=re.M | re.S)
byline_txt = byline.group(0).strip("*") if byline else ""
md = md.replace(byline.group(0), "", 1) if byline else md

body = markdown.markdown(md, extensions=["tables", "sane_lists", "toc"], extension_configs={"toc": {"toc_depth": "2"}})
# Q/H/S/R/C paragraphs -> labelled blocks
body = re.sub(r'<p><strong>(Question|Hypothesis|Setup|Results|Conclusion)\.</strong>',
              lambda m: f'<p class="qh qh-{m.group(1).lower()}"><span class="qlabel">{m.group(1)}</span>', body)
# figures
body = re.sub(r'<p><img alt="([^"]*)" src="([^"]+)" /></p>', r'<figure class="plate"><img loading="lazy" alt="\1" src="\2"></figure>', body)
# tables scroll horizontally
body = body.replace("<table>", '<div class="tablewrap"><table>').replace("</table>", "</table></div>")
# section ids for the TOC come from the toc extension (h2 ids); build a compact nav
toc_items = re.findall(r'<h2 id="([^"]+)">(.*?)</h2>', body)
nav = "\n".join(f'<li><a href="#{i}">{re.sub(r"<[^>]+>", "", t).replace("Experiment ", "").split(" — ")[0]}<span class="navsub">{(re.sub(r"<[^>]+>", "", t).split(" — ")[1] if " — " in t else "")}</span></a></li>' for i, t in toc_items)

html = f'''<meta charset="utf-8">
<title>Gemma Spiral Anatomy</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,700&family=Source+Sans+3:ital,wght@0,400;0,600;1,400&family=JetBrains+Mono:wght@400;500&display=swap">
<style>
:root {{
  --ground: #f7f5f0; --plate: #fcfcfb; --ink: #1b1a17; --muted: #625e56; --rule: #e1ddd3; --accent: #2a78d6; --accent-ink: #1d5aa6;
  --soft: #eef2f8; --code: #edeae2; --qlabel: #7a5a1e; --qbg: #f4efe3;
  --display: "Fraunces", Georgia, "Times New Roman", serif; --body: "Source Sans 3", "Helvetica Neue", Arial, sans-serif; --mono: "JetBrains Mono", Menlo, Consolas, monospace;
}}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{
  color-scheme: dark; --ground: #16181c; --plate: #fcfcfb; --ink: #e9e6df; --muted: #a49f95; --rule: #2d3138; --accent: #7fb3f2; --accent-ink: #a9cbf6;
  --soft: #1e2632; --code: #23262c; --qlabel: #d9b46a; --qbg: #24211a; }} }}
:root[data-theme="dark"] {{
  color-scheme: dark; --ground: #16181c; --plate: #fcfcfb; --ink: #e9e6df; --muted: #a49f95; --rule: #2d3138; --accent: #7fb3f2; --accent-ink: #a9cbf6;
  --soft: #1e2632; --code: #23262c; --qlabel: #d9b46a; --qbg: #24211a; }}
* {{ box-sizing: border-box; }}
body {{ margin: 0; background: var(--ground); color: var(--ink); font-family: var(--body); font-size: 17px; line-height: 1.55; }}
.wrap {{ max-width: 1240px; margin: 0 auto; padding-inline: 20px; padding-block: 28px 80px; display: grid; grid-template-columns: 250px minmax(0, 1fr); gap: 44px; }}
@media (max-width: 1000px) {{ .wrap {{ grid-template-columns: minmax(0, 1fr); gap: 20px; }} nav.toc {{ position: static; max-height: none; }} }}
nav.toc {{ position: sticky; top: env(safe-area-inset-top, 0px); align-self: start; max-height: 100vh; overflow-y: auto; padding-top: 10px; font-size: 14px; }}
nav.toc .eyebrow {{ font-family: var(--mono); font-size: 11px; letter-spacing: .12em; text-transform: uppercase; color: var(--muted); margin: 0 0 10px; }}
nav.toc ol {{ list-style: none; margin: 0; padding: 0; display: flex; flex-direction: column; gap: 6px; }}
nav.toc a {{ color: var(--ink); text-decoration: none; display: block; padding: 4px 8px; border-left: 2px solid var(--rule); }}
nav.toc a:hover, nav.toc a:focus-visible {{ border-left-color: var(--accent); color: var(--accent-ink); outline: none; }}
nav.toc .navsub {{ display: block; color: var(--muted); font-size: 12.5px; line-height: 1.3; }}
main {{ min-width: 0; }}
header.mast {{ border-bottom: 1px solid var(--rule); padding-bottom: 22px; margin-bottom: 28px; }}
header.mast .eyebrow {{ font-family: var(--mono); font-size: 12px; letter-spacing: .12em; text-transform: uppercase; color: var(--accent-ink); margin: 0 0 10px; }}
header.mast h1 {{ font-family: var(--display); font-weight: 700; font-size: clamp(30px, 4.2vw, 46px); line-height: 1.08; margin: 0 0 12px; text-wrap: balance; letter-spacing: -.01em; }}
header.mast p.lede {{ font-size: 19px; color: var(--muted); max-width: 66ch; margin: 0 0 10px; }}
header.mast p.by {{ font-family: var(--mono); font-size: 12.5px; color: var(--muted); margin: 0; }}
h2 {{ font-family: var(--display); font-weight: 700; font-size: 27px; line-height: 1.2; margin: 54px 0 14px; text-wrap: balance; letter-spacing: -.005em; padding-top: 14px; border-top: 1px solid var(--rule); }}
h2:first-of-type {{ border-top: 0; margin-top: 8px; padding-top: 0; }}
h3 {{ font-family: var(--display); font-weight: 500; font-size: 20px; margin: 30px 0 8px; }}
p, li {{ max-width: 72ch; }}
p {{ margin: 0 0 14px; }}
ol, ul {{ padding-left: 1.3em; margin: 0 0 16px; }}
li {{ margin-bottom: 6px; }}
a {{ color: var(--accent-ink); }}
strong {{ font-weight: 600; }}
code {{ font-family: var(--mono); font-size: .86em; background: var(--code); padding: 1px 5px; border-radius: 3px; }}
hr {{ display: none; }}
.qh {{ padding: 10px 14px 10px 14px; border-left: 3px solid var(--qlabel); background: var(--qbg); max-width: none; border-radius: 0 6px 6px 0; }}
.qh .qlabel {{ display: inline-block; font-family: var(--mono); font-size: 11.5px; letter-spacing: .1em; text-transform: uppercase; color: var(--qlabel); margin-right: 10px; }}
.qh-results {{ background: transparent; border-left-color: var(--accent); padding-left: 14px; }}
.qh-results .qlabel {{ color: var(--accent-ink); }}
.qh-conclusion {{ border-left-color: var(--accent); background: var(--soft); }}
.qh-conclusion .qlabel {{ color: var(--accent-ink); }}
figure.plate {{ margin: 18px 0 22px; background: var(--plate); border: 1px solid var(--rule); border-radius: 6px; padding: 10px; }}
figure.plate img {{ display: block; width: 100%; height: auto; max-width: 100%; }}
.tablewrap {{ overflow-x: auto; margin: 6px 0 20px; border: 1px solid var(--rule); border-radius: 6px; }}
table {{ border-collapse: collapse; font-size: 14.5px; font-variant-numeric: tabular-nums; min-width: 100%; }}
th, td {{ padding: 7px 10px; border-bottom: 1px solid var(--rule); text-align: left; vertical-align: top; white-space: nowrap; }}
th {{ font-family: var(--mono); font-size: 11.5px; letter-spacing: .06em; text-transform: uppercase; color: var(--muted); background: var(--soft); }}
tbody tr:last-child td {{ border-bottom: 0; }}
td strong {{ color: var(--accent-ink); }}
blockquote {{ margin: 0 0 16px; padding-left: 14px; border-left: 3px solid var(--rule); color: var(--muted); }}
@media (prefers-reduced-motion: no-preference) {{ html {{ scroll-behavior: smooth; }} }}
</style>
<div class="wrap">
<nav class="toc" aria-label="Sections">
  <p class="eyebrow">Contents</p>
  <ol>{nav}</ol>
</nav>
<main>
<header class="mast">
  <p class="eyebrow">Emotion-concept probes · Gemma 3 27B and Gemma 4 31B</p>
  <h1>What is Gemma 3's distress spiral?</h1>
  <p class="lede">Ten experiments with story-derived emotion and depression vectors, steering, and the assistant axis: what the spiral is, why Gemma 4 does not show it, and how to induce it.</p>
  <p class="by">{byline_txt}</p>
</header>
{body}
</main>
</div>
'''
OUT.write_text(html)
print(f"wrote {OUT} ({OUT.stat().st_size/1024:.0f} KB), {len(toc_items)} sections")
