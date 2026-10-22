import re
from typing import List, Dict, Any

# Resume section header patterns — covers common formats in real PDFs
SECTION_PATTERNS = [
    (r"work\s+experience|employment\s+history|professional\s+experience|experience", "experience"),
    (r"skills?|technical\s+skills?|core\s+competencies|technologies", "skills"),
    (r"projects?|key\s+projects?|notable\s+projects?|portfolio", "project"),
    (r"education|academic\s+background|qualifications?", "education"),
    (r"certifications?|awards?|achievements?|honors?", "general"),
    (r"summary|profile|objective|about\s+me|professional\s+summary", "general"),
    (r"contact|personal\s+information|personal\s+details", "general"),
    (r"internship|training", "experience"),
    (r"publications?|research", "general"),
]

def _detect_category_from_text(text: str, source_name: str) -> tuple[str, str]:
    """
    Returns (category, detected_header) by scanning for known resume section headers.
    Priority: match against known section patterns before falling back to source name.
    """
    lower = text.lower()
    lower_src = source_name.lower()

    for pattern, cat in SECTION_PATTERNS:
        if re.search(pattern, lower):
            # Use the first matching line as the header label
            for line in text.split("\n"):
                if re.search(pattern, line.lower()):
                    return cat, line.strip()[:80]
            return cat, cat.title()

    # Fallback: derive from source filename
    if "project" in lower_src:
        return "project", "Projects"
    if "skill" in lower_src:
        return "skills", "Skills"
    return "general", "General"


class DocumentChunker:
    """Splits resume documents into semantic, section-aware chunks for RAG retrieval."""

    @staticmethod
    def chunk_markdown(content: str, source_name: str) -> List[Dict[str, Any]]:
        chunks = []
        # Split by level 2 or level 3 markdown headers
        sections = re.split(r'\n(?=#{2,}\s)', content)

        for idx, section in enumerate(sections):
            cleaned = section.strip()
            if not cleaned:
                continue

            # Extract header if present
            header_match = re.match(r'^(#{2,})\s+(.+)', cleaned)
            header_title = header_match.group(2).strip() if header_match else "General"

            # Determine category from header title
            category, _ = _detect_category_from_text(header_title, source_name)

            chunks.append({
                "id": f"{source_name}_{idx}",
                "text": cleaned,
                "metadata": {
                    "source": source_name,
                    "header": header_title,
                    "category": category,
                }
            })

        return chunks

    @staticmethod
    def chunk_pdf(file_path: str, source_name: str) -> List[Dict[str, Any]]:
        """
        Extracts text from a resume PDF using section-boundary detection.
        Splits on known resume section headers so each chunk carries meaningful context.
        """
        chunks = []
        try:
            from pypdf import PdfReader
        except ImportError:
            print(f"[DocumentChunker] 'pypdf' not installed. Run: pip install pypdf")
            return []

        try:
            reader = PdfReader(file_path)

            # Step 1: Extract and clean full document text with page markers
            full_text_lines: List[tuple[int, str]] = []  # (page_num, line)
            for page_num, page in enumerate(reader.pages, start=1):
                raw = page.extract_text() or ""
                raw = re.sub(r'[\x00\r]', '', raw)
                raw = re.sub(r'[ \t]+', ' ', raw)
                for line in raw.split('\n'):
                    stripped = line.strip()
                    if stripped:
                        full_text_lines.append((page_num, stripped))

            if not full_text_lines:
                print(f"[DocumentChunker] No text extracted from '{source_name}'")
                return []

            # Step 2: Identify section boundaries
            # A line is a section header if it's short, ALL-CAPS or Title Case,
            # OR matches known resume section keywords
            HEADER_RE = re.compile(
                r'^(' +
                r'work\s+experience|experience|employment|professional\s+experience|'
                r'skills?|technical\s+skills?|core\s+competencies|technologies|'
                r'education|academic|qualification|'
                r'projects?|portfolio|'
                r'certifications?|awards?|achievements?|'
                r'summary|profile|objective|about|'
                r'contact|personal\s+info|'
                r'internship|training|'
                r'publications?|research|'
                r'languages?|tools?'
                r')[\s:]*$',
                re.IGNORECASE
            )

            def _is_section_header(line: str) -> bool:
                stripped = line.strip()
                if len(stripped) > 60:
                    return False
                # Exact keyword match
                if HEADER_RE.match(stripped):
                    return True
                # All-caps short line (common PDF header format)
                if stripped.isupper() and 3 <= len(stripped) <= 50:
                    return True
                return False

            # Step 3: Split into sections
            sections: List[Dict] = []
            current_header = "Profile"
            current_page = full_text_lines[0][0] if full_text_lines else 1
            current_lines: List[str] = []

            for page_num, line in full_text_lines:
                if _is_section_header(line):
                    # Save accumulated lines as a section
                    if current_lines:
                        sections.append({
                            "header": current_header,
                            "page": current_page,
                            "text": "\n".join(current_lines),
                        })
                    current_header = line.strip()
                    current_page = page_num
                    current_lines = []
                else:
                    current_lines.append(line)

            # Flush last section
            if current_lines:
                sections.append({
                    "header": current_header,
                    "page": current_page,
                    "text": "\n".join(current_lines),
                })

            # Step 4: If no section headers found, fall back to page-based chunking
            if len(sections) <= 1:
                print(f"[DocumentChunker] No section headers detected in '{source_name}', using page-based chunking.")
                sections = []
                for page_num, page in enumerate(reader.pages, start=1):
                    raw = page.extract_text() or ""
                    raw = re.sub(r'[\x00\r]', '', raw)
                    raw = re.sub(r'[ \t]+', ' ', raw).strip()
                    if raw:
                        sections.append({
                            "header": f"Page {page_num}",
                            "page": page_num,
                            "text": raw,
                        })

            # Step 5: Convert sections to chunks, splitting large ones
            MAX_CHUNK_CHARS = 800

            for sec_idx, sec in enumerate(sections):
                header = sec["header"]
                page = sec["page"]
                text = sec["text"].strip()

                if not text or len(text) < 20:
                    continue

                category, _ = _detect_category_from_text(header + " " + text[:200], source_name)

                if len(text) <= MAX_CHUNK_CHARS:
                    # Prepend section header into the text so LLM always sees context
                    chunk_text = f"[Section: {header}]\n{text}"
                    chunks.append({
                        "id": f"{source_name}_s{sec_idx}",
                        "text": chunk_text,
                        "metadata": {
                            "source": source_name,
                            "page": page,
                            "header": header,
                            "category": category,
                        }
                    })
                else:
                    # Split large sections into sub-chunks, preserving header context
                    lines = text.split('\n')
                    buffer: List[str] = []
                    buf_len = 0
                    sub_idx = 0

                    for line in lines:
                        buffer.append(line)
                        buf_len += len(line)
                        if buf_len >= MAX_CHUNK_CHARS:
                            chunk_text = f"[Section: {header}]\n" + "\n".join(buffer)
                            chunks.append({
                                "id": f"{source_name}_s{sec_idx}_{sub_idx}",
                                "text": chunk_text,
                                "metadata": {
                                    "source": source_name,
                                    "page": page,
                                    "header": header,
                                    "category": category,
                                }
                            })
                            buffer = []
                            buf_len = 0
                            sub_idx += 1

                    if buffer:
                        chunk_text = f"[Section: {header}]\n" + "\n".join(buffer)
                        chunks.append({
                            "id": f"{source_name}_s{sec_idx}_{sub_idx}",
                            "text": chunk_text,
                            "metadata": {
                                "source": source_name,
                                "page": page,
                                "header": header,
                                "category": category,
                            }
                        })

            print(f"[DocumentChunker] '{source_name}': {len(chunks)} chunks from {len(sections)} sections.")

        except Exception as e:
            print(f"[DocumentChunker] Error parsing PDF '{source_name}': {e}")

        return chunks
