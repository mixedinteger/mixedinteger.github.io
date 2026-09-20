#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."

build_page() {
  local source="$1"
  local output="$2"
  local active="$3"

  pandoc "2027/competition/${source}" \
    -f gfm+tex_math_dollars \
    -t html5 \
    --standalone \
    --metadata title="MIPcc27: The 2027 Land-Doig MIP Competition" \
    --mathjax=https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js \
    --css=../css/normalize.css \
    --css=../css/skeleton.css \
    --css=../css/colors.css \
    --css=../css/site.css \
    -o "2027/competition/${output}"

  # Insert the workshop-style page header and footer around the generated content.
  python3 - "${output}" "${active}" <<'PY'
from pathlib import Path
import re
import sys
root = Path.cwd()
out = root / '2027/competition' / sys.argv[1]
active = sys.argv[2]
header = (root / '2027/competition/page-header.html').read_text()
for page in ('home', 'topic', 'rules'):
    header = header.replace(f'__ACTIVE_{page.upper()}__', 'active' if page == active else '')
header = header.replace(' class=""', '')
footer = (root / '2027/competition/page-footer.html').read_text()
content = out.read_text()
content = re.sub(r'\n<header id="title-block-header">.*?</header>\n', '\n', content, flags=re.S)
body = content.split('<body>', 1)[1].split('</body>', 1)[0]
body = body.strip()
pattern = re.compile(r'<h2\b[^>]*>.*?</h2>', re.S)
matches = list(pattern.finditer(body))
if matches:
    sections = []
    section_classes = ['mainstyle1', 'mainstyle2']
    intro = body[:matches[0].start()].strip()
    if intro:
        sections.append(f'<section class="highlight">{intro}</section>')
    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(body)
        block = body[start:end].strip()
        if block:
            sections.append(f'<section class="{section_classes[i % len(section_classes)]}">{block}</section>')
    body = '\n'.join(sections)
new_html = content.split('<body>', 1)[0] + '<body>' + '\n' + header + '\n' + body + '\n' + footer + '\n</body>\n</html>\n'
out.write_text(new_html)
PY
}

build_page _index.md index.html home
build_page _topic.md topic.html topic
build_page _rules.md rules.html rules
