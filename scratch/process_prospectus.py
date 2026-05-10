
import pypdf
import os
import re

pdf_path = r"c:\Users\carlos\OneDrive - Vaal University of Technology\WORK\2026\AI_v2_html\Books\Prospectus 2026 FET  FINAL_20 Jan 2026.pdf"
output_path = r"c:\Users\carlos\OneDrive - Vaal University of Technology\WORK\2026\AI_v2_html\Week 10 - Generative AI & Large Language Models\vut_prospectus_text.txt"

print(f"Reading {pdf_path}...")

try:
    reader = pypdf.PdfReader(pdf_path)
    full_text = ""
    
    for i in range(len(reader.pages)):
         page = reader.pages[i]
         text = page.extract_text()
         if text:
             # Basic cleaning: remove extra whitespace and headers
             text = re.sub(r'\s+', ' ', text)
             full_text += text + "\n"
    
    # Save to text file
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(full_text)
        
    print(f"Successfully extracted {len(full_text)} characters to {output_path}")
    
except Exception as e:
    print(f"Error reading PDF: {e}")
