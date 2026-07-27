from pathlib import Path
import PyPDF2

p = Path('lessons/02-supervision.pdf')
reader = PyPDF2.PdfReader(str(p))
terms = ['decision tree', 'tree', 'regression', 'log', 'supervised', 'classification', 'entropy', 'gini']
for i, page in enumerate(reader.pages):
    text = page.extract_text() or ''
    lower = text.lower()
    hits = [t for t in terms if t in lower]
    if hits:
        print('PAGE', i + 1, 'hits:', hits)
        print(text[:2000])
        print('---')
