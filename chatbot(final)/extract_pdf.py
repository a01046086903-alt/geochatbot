import fitz
import sys
from collections import Counter
import re

def extract_markdown(pdf_path, md_path):
    doc = fitz.open(pdf_path)
    md_lines = []
    
    # First pass: find the most common font size to identify body text
    sizes = []
    for page in doc:
        blocks = page.get_text("dict")["blocks"]
        for b in blocks:
            if b['type'] == 0:  # text block
                for l in b["lines"]:
                    for s in l["spans"]:
                        text = s["text"].strip()
                        if text:
                            # round size to 1 decimal point for grouping
                            sizes.append(round(s["size"], 1))
                            
    if not sizes:
        print("No text found in the PDF.")
        return
        
    # The most frequent font size is assumed to be the normal body text size
    body_size = Counter(sizes).most_common(1)[0][0]
    print(f"Detected body font size: {body_size}")
    
    # Second pass: extract text and apply markdown formatting based on font size
    for page in doc:
        blocks = page.get_text("dict")["blocks"]
        for b in blocks:
            if b['type'] == 0:  # text
                block_text = ""
                is_header = False
                
                for l in b["lines"]:
                    for s in l["spans"]:
                        text = s["text"]
                        if not text.strip():
                            block_text += text
                            continue
                            
                        font_size = round(s["size"], 1)
                        # If font is significantly larger than body text, treat as header
                        if font_size > body_size + 0.5:
                            is_header = True
                            
                        block_text += text
                
                # Combine lines in the block to remove unnatural line breaks
                # Many PDF blocks represent a single paragraph
                # But keep bullet point marker at the beginning
                is_bullet = False
                block_text = block_text.strip()
                if block_text.startswith("•") or block_text.startswith("- "):
                    is_bullet = True
                    block_text = block_text[1:].strip()
                
                block_text = block_text.replace('\n', ' ')
                # Remove extra multiple spaces that could have resulted from joining
                block_text = re.sub(r'\s+', ' ', block_text)
                
                if block_text:
                    # Also treat (가), (나) etc as subheaders or bold text
                    if re.match(r'^\([가-힣]\)\s', block_text):
                        md_lines.append(f"\n### {block_text}\n")
                    elif is_header:
                        # Convert to markdown header (H2)
                        md_lines.append(f"\n## {block_text}\n")
                    elif is_bullet:
                        md_lines.append(f"* {block_text}")
                    else:
                        md_lines.append(block_text)
                
                # Add paragraph separation after each block
                md_lines.append("")

    # Clean up multiple newlines
    final_text = "\n".join(md_lines)
    final_text = re.sub(r'\n{3,}', '\n\n', final_text)

    with open(md_path, "w", encoding="utf-8") as f:
        f.write(final_text.strip() + "\n")
        
    print(f"Successfully extracted markdown to {md_path}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python extract_pdf.py <input_pdf> <output_md>")
        sys.exit(1)
    
    input_pdf = sys.argv[1]
    output_md = sys.argv[2]
    extract_markdown(input_pdf, output_md)
