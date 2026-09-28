"""Quick EDA to explore the PDF"""
from pypdf import PdfReader

pdf_path = "introduction to internal combustion engines.pdf"
reader = PdfReader(pdf_path)

print("="*80)
print("PDF EXPLORATORY DATA ANALYSIS")
print("="*80)
print(f"\nTotal Pages: {len(reader.pages)}")

# Extract all text
full_text = ""
for page in reader.pages:
    full_text += page.extract_text() + "\n"

print(f"Total Characters: {len(full_text)}")
print(f"Total Words (approx): {len(full_text.split())}")

print("\n" + "="*80)
print("FIRST 1000 CHARACTERS")
print("="*80)
print(full_text[:1000])

print("\n" + "="*80)
print("SAMPLE FROM MIDDLE")
print("="*80)
mid_point = len(full_text) // 2
print(full_text[mid_point:mid_point+500])

