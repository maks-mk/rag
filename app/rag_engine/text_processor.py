import re
import io
from .config import (
    CHARS_PER_TOKEN,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    MIN_CHUNK_CHARS,
)

def count_tokens(text: str) -> int:
    return int(len(text) / CHARS_PER_TOKEN)


# ── Text extraction ───────────────────────────────────────────────────────────

def extract_txt(content: bytes) -> str:
    for enc in ("utf-8", "utf-8-sig", "cp1251", "latin-1"):
        try:
            return content.decode(enc)
        except UnicodeDecodeError:
            continue
    return content.decode("utf-8", errors="replace")

def extract_pdf(content: bytes) -> str:
    try:
        import pdfplumber
        parts = []
        with pdfplumber.open(io.BytesIO(content)) as pdf:
            for page in pdf.pages:
                t = page.extract_text()
                if t:
                    parts.append(t)
        return "\n\n".join(parts)
    except ImportError:
        pass
    
    from pypdf import PdfReader
    reader = PdfReader(io.BytesIO(content))
    parts = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            parts.append(text)
    return "\n\n".join(parts)

def extract_docx(content: bytes) -> str:
    from docx import Document
    doc = Document(io.BytesIO(content))
    parts = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
    for table in doc.tables:
        for row in table.rows:
            row_text = " | ".join(c.text.strip() for c in row.cells if c.text.strip())
            if row_text:
                parts.append(row_text)
    return "\n\n".join(parts)

def extract_text(content: bytes, ext: str, filename: str) -> str:
    if ext in (".txt", ".md"):   return extract_txt(content)
    if ext == ".pdf":            return extract_pdf(content)
    if ext == ".docx":           return extract_docx(content)
    raise ValueError(f"Unsupported extension: {ext}")


# ── Chunking ──────────────────────────────────────────────────────────────────

def chunk_text(text: str, chunk_tokens: int = CHUNK_SIZE, overlap_tokens: int = CHUNK_OVERLAP) -> list[str]:
    chunk_chars   = chunk_tokens * 4
    overlap_chars = overlap_tokens * 4

    text = re.sub(r"\n{3,}", "\n\n", text)
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    chunks: list[str] = []
    current = ""

    for para in paragraphs:
        if len(para) > chunk_chars:
            sentences = [s.strip() for s in para.split("\n") if s.strip()]
            for sentence in sentences:
                if len(sentence) > chunk_chars:
                    for i in range(0, len(sentence), chunk_chars - overlap_chars):
                        part = sentence[i : i + chunk_chars]
                        if len(current) + len(part) > chunk_chars and current:
                            chunks.append(current)
                            current = part
                        else:
                            current = (current + " " + part).strip()
                    continue
                
                if len(current) + len(sentence) + 2 <= chunk_chars:
                    current = (current + "\n" + sentence).strip()
                else:
                    if current and len(current) >= MIN_CHUNK_CHARS:
                        chunks.append(current)
                    overlap_text = current[-overlap_chars:] if len(current) > overlap_chars else current
                    current = (overlap_text + "\n" + sentence).strip()
        else:
            if len(current) + len(para) + 2 <= chunk_chars:
                current = (current + "\n\n" + para).strip()
            else:
                if current and len(current) >= MIN_CHUNK_CHARS:
                    chunks.append(current)
                overlap_text = current[-overlap_chars:] if len(current) > overlap_chars else current
                current = (overlap_text + "\n\n" + para).strip()

    if current and len(current) >= MIN_CHUNK_CHARS:
        chunks.append(current)

    merged: list[str] = []
    for c in chunks:
        if merged and len(c) < MIN_CHUNK_CHARS:
            merged[-1] += "\n\n" + c
        else:
            merged.append(c)

    return merged
