import re
from typing import List, Dict, Any

class DocumentChunker:
    """Splits markdown documents into semantic, header-aware chunks."""

    @staticmethod
    def chunk_markdown(content: str, source_name: str) -> List[Dict[str, Any]]:
        chunks = []
        # Split by level 2 or level 3 markdown headers
        sections = re.split(r'\n(?=##+\s)', content)

        for idx, section in enumerate(sections):
            cleaned = section.strip()
            if not cleaned:
                continue

            # Extract header if present
            header_match = re.match(r'^(##+)\s+(.+)', cleaned)
            header_title = header_match.group(2).strip() if header_match else "General"

            # Determine semantic category based on header or source filename
            category = "general"
            lower_h = header_title.lower()
            lower_src = source_name.lower()
            if "project" in lower_h or "project" in lower_src:
                category = "project"
            elif "experience" in lower_h or "work" in lower_h:
                category = "experience"
            elif "skill" in lower_h or "skill" in lower_src:
                category = "skills"
            elif "education" in lower_h:
                category = "education"

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
