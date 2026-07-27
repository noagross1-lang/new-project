from pathlib import Path
import PyPDF2

p = Path('lessons/02-supervision.pdf')
print('exists', p.exists())
reader = PyPDF2.PdfReader(str(p))
print('pages', len(reader.pages))
for i in range(min(5, len(reader.pages))):
    text = reader.pages[i].extract_text()
    print('--- PAGE', i + 1, '---')
    if text is None:
        print('(no text)')
    else:
        print(text[:1600])
