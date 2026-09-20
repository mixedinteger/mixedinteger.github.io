#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."

pandoc 2027/competition/_README.md \
  -f gfm+tex_math_dollars \
  -t html5 \
  --standalone \
  --metadata title="MIPcc27: The 2027 Land-Doig MIP Competition" \
  --mathjax=https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js \
  --css=../css/normalize.css \
  --css=../css/skeleton.css \
  --css=../css/colors.css \
  --css=../css/site.css \
  -o 2027/competition/index.html

# Insert the workshop-style page header and footer around the generated content.
python3 - <<'PY'
from pathlib import Path
import re
root = Path('/home/pierre/git/mixedinteger.github.io')
out = root / '2027/competition/index.html'
header = (root / '2027/competition/page-header.html').read_text()
footer = (root / '2027/competition/page-footer.html').read_text()
content = out.read_text()
content = re.sub(r'\n<header id="title-block-header">.*?</header>\n', '\n', content, flags=re.S)
content = re.sub(r'\n<img src="https://www\.mixedinteger\.org/2027/images/mip-2027\.jpg" width="400">\n', '\n', content)
content = re.sub(r'\n<h1 id="mipcc27-the-2027-land-doig-mip-competition">.*?</h1>\n', '\n', content, count=1, flags=re.S)
body = content.split('<body>', 1)[1].split('</body>', 1)[0]
body = body.strip()
pattern = re.compile(r'<h2\b[^>]*>.*?</h2>', re.S)
matches = list(pattern.finditer(body))
if matches:
    sections = []
    classes = ['highlight', 'mainstyle2', 'mainstyle1']
    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(body)
        block = body[start:end].strip()
        if block:
          class_name = 'mainstyle2' if 'id="references"' in match.group(0) else classes[i % len(classes)]
          sections.append(f'<section class="{class_name}">{block}</section>')
    body = '\n'.join(sections)
new_html = content.split('<body>', 1)[0] + '<body>' + '\n' + header + '\n' + body + '\n' + footer + '\n</body>\n</html>\n'
out.write_text(new_html)
PY
