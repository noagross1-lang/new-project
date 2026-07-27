import re
import sys
import markdown

SRC = "writeup.md"
NO_IMAGES = "--no-images" in sys.argv
HTML_OUT = "writeup_noimages.html" if NO_IMAGES else "writeup.html"

with open(SRC, encoding="utf-8") as f:
    md_text = f.read()

if NO_IMAGES:
    md_text = re.sub(r"!\[[^\]]*\]\([^)]*\)\n?", "", md_text)

html_body = markdown.markdown(md_text, extensions=["tables", "fenced_code", "nl2br"])

# Turn leading "**Label:**" paragraphs into styled block labels (Solution/
# Evaluation Criteria/Setup/Results/Visualization/Impediments pattern).
html_body = re.sub(
    r"<p><strong>([^<]+?)</strong>",
    r'<p><span class="field-label">\1</span>',
    html_body,
)

# Wrap consecutive images in a figure block for nicer spacing/border.
html_body = re.sub(
    r"<p>(<img [^>]+>)</p>",
    r'<div class="figure">\1</div>',
    html_body,
)

# Split off the title + team section as the cover page; everything from
# "Domain" onward is the regular flowing content.
marker = "<h2>דומיין"
idx = html_body.find(marker)
pre_cover = html_body[:idx]
rest_html = html_body[idx:]

title_match = re.search(r"<h1>(.*?)</h1>", pre_cover, re.S)
tagline_match = re.search(r"<p><em>(.*?)</em></p>", pre_cover, re.S)
team_list_match = re.search(r"<h2>צוות</h2>\s*(<ul>.*?</ul>)", pre_cover, re.S)

title_text = title_match.group(1) if title_match else ""
tagline_text = tagline_match.group(1) if tagline_match else ""
team_list_html = team_list_match.group(1) if team_list_match else ""

cover_html = f"""
<div class="cover">
  <div class="kicker">פרויקט גמר &middot; ניתוח מאגרי מידע</div>
  <h1>{title_text}</h1>
  <div class="divider"></div>
  <div class="tagline">{tagline_text}</div>
  <div class="team-box">
    <h3>צוות הפרויקט</h3>
    {team_list_html}
  </div>
</div>
"""

PAGE = f"""<!doctype html>
<html lang="he" dir="rtl">
<head>
<meta charset="utf-8">
<title>Finding Hidden Gems in the Rome Airbnb Market</title>
<style>
  @page {{
    size: A4;
    margin: 15mm 16mm 15mm 16mm;
  }}
  * {{ box-sizing: border-box; }}
  html, body {{
    direction: rtl;
    text-align: right;
    font-family: "Segoe UI", "Assistant", Arial, sans-serif;
    color: #1c1c1c;
    line-height: 1.42;
    font-size: 10.3pt;
  }}
  .cover {{
    text-align: center;
    padding: 38mm 8mm 14mm 8mm;
    page-break-after: always;
  }}
  .cover .kicker {{
    color: #0072B2;
    letter-spacing: 2px;
    font-weight: 600;
    font-size: 11pt;
    margin-bottom: 10mm;
  }}
  .cover h1 {{
    font-size: 27pt;
    line-height: 1.35;
    color: #0b3d4d;
    margin: 0 0 6mm 0;
    border: none;
    padding: 0;
  }}
  .cover .tagline {{
    font-size: 13pt;
    color: #555;
    font-style: italic;
    margin-bottom: 18mm;
  }}
  .cover .divider {{
    width: 70mm;
    height: 3px;
    background: linear-gradient(90deg, #0072B2, #E69F00, #009E73);
    margin: 0 auto 16mm auto;
    border-radius: 2px;
  }}
  .team-box {{
    display: inline-block;
    text-align: right;
    background: #f4f8fa;
    border: 1px solid #d7e3e8;
    border-radius: 8px;
    padding: 6mm 10mm;
    margin-top: 6mm;
    min-width: 90mm;
  }}
  .team-box h3 {{
    margin: 0 0 3mm 0;
    color: #0072B2;
    font-size: 11pt;
  }}
  .team-box ul {{ margin: 0; padding-inline-start: 5mm; }}
  .team-box li {{ margin-bottom: 1.5mm; }}
  .cover .draft-note {{
    margin-top: 14mm;
    font-size: 9pt;
    color: #999;
    max-width: 130mm;
    margin-inline: auto;
  }}

  h2 {{
    color: #0b3d4d;
    font-size: 14pt;
    border-bottom: 2px solid #0072B2;
    padding-bottom: 1.3mm;
    margin-top: 6mm;
    margin-bottom: 3mm;
    page-break-after: avoid;
  }}
  h2:first-of-type {{ margin-top: 0; }}

  p {{ margin: 0 0 2mm 0; text-align: justify; }}
  strong {{ color: #14313a; }}

  .field-label {{
    display: block;
    color: #0072B2;
    font-weight: 700;
    font-size: 9.6pt;
    letter-spacing: 0.3px;
    margin-bottom: 0.5mm;
  }}

  ul, ol {{ margin: 0 0 2mm 0; padding-inline-start: 6mm; }}
  li {{ margin-bottom: 0.6mm; text-align: justify; }}

  table {{
    border-collapse: collapse;
    width: 100%;
    margin: 2.5mm 0 3mm 0;
    font-size: 9pt;
    page-break-inside: avoid;
  }}
  th, td {{
    border: 1px solid #cdd8dc;
    padding: 1.2mm 2mm;
    text-align: center;
  }}
  th {{
    background: #0072B2;
    color: white;
    font-weight: 600;
  }}
  tr:nth-child(even) td {{ background: #f4f8fa; }}

  .figure {{
    text-align: center;
    margin: 3mm 0;
    page-break-inside: avoid;
  }}
  .figure img {{
    max-width: 92%;
    max-height: 78mm;
    border: 1px solid #d7e3e8;
    border-radius: 4px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.08);
  }}

  code {{
    background: #f0f4f5;
    border-radius: 3px;
    padding: 0.5px 4px;
    font-family: "Consolas", monospace;
    font-size: 92%;
    direction: ltr;
    display: inline-block;
  }}

  em {{ color: #555; }}

  hr {{ border: none; border-top: 1px solid #d7e3e8; margin: 6mm 0; }}
</style>
</head>
<body>
{cover_html}
<div class="content">
{rest_html}
</div>
</body>
</html>
"""

with open(HTML_OUT, "w", encoding="utf-8") as f:
    f.write(PAGE)

print(f"Wrote {HTML_OUT}")
