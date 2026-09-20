"""Insert the workshop-style page header and footer around pandoc-generated content.

Usage: python3 wrap_page.py <output.html> <active-page>
"""

from pathlib import Path
import re
import sys

root = Path(__file__).resolve().parent
out = Path(sys.argv[1])
active = sys.argv[2]

header = (root / "page-header.html").read_text()
for page in ("home", "topic", "rules"):
    header = header.replace(
        f"__ACTIVE_{page.upper()}__", "active" if page == active else ""
    )
header = header.replace(' class=""', "")

footer = (root / "page-footer.html").read_text()

content = out.read_text()
content = re.sub(
    r'\n<header id="title-block-header">.*?</header>\n', "\n", content, flags=re.S
)
head, rest = content.split("<body>", 1)
body, _tail = rest.split("</body>", 1)
body = body.strip()

pattern = re.compile(r"<h2\b[^>]*>.*?</h2>", re.S)
matches = list(pattern.finditer(body))
if matches:
    sections = []
    section_classes = ["mainstyle1", "mainstyle2"]
    intro = body[: matches[0].start()].strip()
    if intro:
        sections.append(f'<section class="highlight">{intro}</section>')
    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(body)
        block = body[start:end].strip()
        if block:
            sections.append(
                f'<section class="{section_classes[i % len(section_classes)]}">{block}</section>'
            )
    body = "\n".join(sections)

new_html = (
    head
    + "<body>"
    + "\n"
    + header
    + "\n"
    + body
    + "\n"
    + footer
    + "\n</body>\n</html>\n"
)
out.write_text(new_html)
