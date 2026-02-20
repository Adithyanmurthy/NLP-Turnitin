#!/usr/bin/env python3
"""Convert Literature_Survey.docx to PDF using subprocess and macOS tools."""
import subprocess
import sys
import os

docx_path = os.path.abspath("Literature_Survey.docx")
pdf_path = os.path.abspath("Literature_Survey.pdf")

# Try method 1: docx2pdf (if installed)
try:
    from docx2pdf import convert
    convert(docx_path, pdf_path)
    print(f"PDF saved to: {pdf_path}")
    sys.exit(0)
except ImportError:
    print("docx2pdf not installed, trying other methods...")
except Exception as e:
    print(f"docx2pdf failed: {e}")

# Try method 2: LibreOffice (if installed)
for lo_path in [
    "/Applications/LibreOffice.app/Contents/MacOS/soffice",
    "/usr/local/bin/soffice",
    "soffice",
]:
    try:
        result = subprocess.run(
            [lo_path, "--headless", "--convert-to", "pdf", "--outdir", ".", docx_path],
            capture_output=True, text=True, timeout=60
        )
        if os.path.exists(pdf_path):
            print(f"PDF saved to: {pdf_path}")
            sys.exit(0)
        else:
            print(f"LibreOffice ran but PDF not found: {result.stderr}")
    except (FileNotFoundError, subprocess.TimeoutExpired):
        continue

# Try method 3: textutil + cupsfilter (macOS built-in, limited formatting)
print("\nNo PDF converter found. Please install one:")
print("  Option 1: pip install docx2pdf  (requires Microsoft Word installed)")
print("  Option 2: brew install --cask libreoffice")
print(f"\nThen run: /usr/local/bin/python3 convert_to_pdf.py")
print(f"\nOr open {docx_path} in Word/Pages and export as PDF manually.")
