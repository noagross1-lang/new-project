"""Builds write_up_problem_4.docx from write_up_problem_4_source.md, reusing
the same RTL-fixed rendering logic as build_docx.py (Hebrew font, bidi
section/style/paragraph properties, table/heading/image styling) so the new
document matches writeup.docx's look without duplicating that code.
"""
import build_docx

build_docx.SRC = "write_up_problem_4_source.md"
build_docx.OUT = "write_up_problem_4.docx"

if __name__ == "__main__":
    build_docx.build()
