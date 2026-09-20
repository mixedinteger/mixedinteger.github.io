#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

build_page() {
  local source="$1"
  local output="$2"
  local active="$3"

  pandoc "${source}" \
    -f gfm+tex_math_dollars \
    -t html5 \
    --standalone \
    --metadata title="The 2027 Land-Doig MIP Competition" \
    --mathjax=https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js \
    --css=https://www.mixedinteger.org/2027/css/normalize.css \
    --css=https://www.mixedinteger.org/2027/css/skeleton.css \
    --css=https://www.mixedinteger.org/2027/css/colors.css \
    --css=https://www.mixedinteger.org/2027/css/site.css \
    -o "${output}"

  # Insert the workshop-style page header and footer around the generated content.
  python3 wrap_page.py "${output}" "${active}"
}

build_page _index.md index.html home
build_page _topic.md topic.html topic
build_page _rules.md rules.html rules
