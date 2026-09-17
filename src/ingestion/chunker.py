from dataclasses import dataclass
from typing import Any, List, Optional


@dataclass
class TextChunk:
    text: str
    chunk_index: int
    page: int
    source: str
    char_count: int
    estimated_tokens: int
    section: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "chunk_index": self.chunk_index,
            "page": self.page,
            "source": self.source,
            "char_count": self.char_count,
            "estimated_tokens": self.estimated_tokens,
            "section": self.section,
        }



class RecursiveTextSplitter:
    """
    Custom recursive text splitter that splits text hierarchically on paragraph,
    Completely independent of external frameworks.
    """

    def __init__(
        self,
        chunk_size: int = 800,
        chunk_overlap: int = 150,
        separators: Optional[List[str]] = None,
    ):
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be strictly smaller than chunk_size.")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", ". ", "? ", "! ", " ", ""]

    def _split_text_recursively(self, text: str, separators: List[str]) -> List[str]:
        """Recursively break text down by separators until all parts fit within chunk_size."""
        if not text.strip():
            return []

        # If text already fits, return it
        if len(text) <= self.chunk_size:
            return [text.strip()]

        # Pick the first separator present in the text
        selected_sep = ""
        remaining_seps: List[str] = []
        for i, sep in enumerate(separators):
            if sep == "":
                selected_sep = ""
                remaining_seps = []
                break
            if sep in text:
                selected_sep = sep
                remaining_seps = separators[i + 1:]
                break

        # Fallback to hard character slicing if no separator remains
        if selected_sep == "":
            return self._hard_slice_by_chars(text)

        # Split text by the selected separator
        splits = text.split(selected_sep)
        good_splits: List[str] = []

        for s in splits:
            s_clean = s.strip()
            if not s_clean:
                continue
            if len(s_clean) <= self.chunk_size:
                good_splits.append(s_clean)
            else:
                # Recurse with finer separators
                sub_splits = self._split_text_recursively(s_clean, remaining_seps)
                good_splits.extend(sub_splits)

        # Merge splits with sliding window overlap
        join_sep = " " if selected_sep in [" ", ""] else "\n\n"
        return self._merge_splits(good_splits, separator=join_sep)

    def _hard_slice_by_chars(self, text: str) -> List[str]:
        """Hard character slice when no natural boundary exists."""
        chunks = []
        step = max(1, self.chunk_size - self.chunk_overlap)
        for i in range(0, len(text), step):
            chunk = text[i : i + self.chunk_size].strip()
            if chunk:
                chunks.append(chunk)
        return chunks

    def _merge_splits(self, splits: List[str], separator: str) -> List[str]:
        """Merge atomic splits into chunks with sliding overlap."""
        merged_chunks: List[str] = []
        current_chunk_parts: List[str] = []
        current_len = 0

        for s in splits:
            part = s.strip()
            if not part:
                continue
            part_len = len(part)

            # Test if adding this part exceeds chunk_size
            test_len = current_len + part_len + (len(separator) if current_chunk_parts else 0)
            if test_len <= self.chunk_size:
                current_chunk_parts.append(part)
                current_len = test_len
            else:
                if current_chunk_parts:
                    chunk_str = separator.join(current_chunk_parts).strip()
                    if chunk_str:
                        merged_chunks.append(chunk_str)

                    # Build overlap from the tail of current parts
                    overlap_parts: List[str] = []
                    overlap_len = 0
                    for rev_part in reversed(current_chunk_parts):
                        if overlap_len + len(rev_part) <= self.chunk_overlap:
                            overlap_parts.insert(0, rev_part)
                            overlap_len += len(rev_part) + len(separator)
                        else:
                            break

                    current_chunk_parts = overlap_parts + [part]
                    current_len = len(separator.join(current_chunk_parts))
                else:
                    merged_chunks.append(part)
                    current_chunk_parts = []
                    current_len = 0

        if current_chunk_parts:
            final_str = separator.join(current_chunk_parts).strip()
            if final_str:
                merged_chunks.append(final_str)

        return merged_chunks

    def split_pages(self, pages: List[dict[str, Any]]) -> List[TextChunk]:
        """
        Split a list of extracted page dictionaries:
        [{'page': int, 'text': str, 'source': str}, ...]
        into a list of TextChunk objects with preserved metadata.
        """
        all_chunks: List[TextChunk] = []
        global_idx = 0

        for page_data in pages:
            raw_text = page_data.get("text", "")
            page_num = page_data.get("page", 1)
            source = page_data.get("source", "unknown")

            raw_chunks = self._split_text_recursively(raw_text, self.separators)

            for text_chunk in raw_chunks:
                clean_chunk = text_chunk.strip()
                if not clean_chunk:
                    continue
                all_chunks.append(
                    TextChunk(
                        text=clean_chunk,
                        chunk_index=global_idx,
                        page=page_num,
                        source=source,
                        char_count=len(clean_chunk),
                        estimated_tokens=max(1, len(clean_chunk) // 4),
                    )
                )
                global_idx += 1

        return all_chunks


class QAStructureSplitter:
    """
    Structure-aware chunker designed for Q&A documents (e.g. IELTS Spirit package FAQs).
    Groups content by course package sections and pairs questions with their answers,
    attaching the package header as context metadata to ensure clean semantic retrieval.
    """

    def __init__(self, max_chunk_size: int = 800, fallback_splitter: Optional[RecursiveTextSplitter] = None):
        self.max_chunk_size = max_chunk_size
        self.fallback_splitter = fallback_splitter or RecursiveTextSplitter(chunk_size=max_chunk_size)

    def _is_section_header(self, line: str) -> bool:
        clean = line.strip()
        import re
        if re.match(r'^[১-৯\d]+\.\s*(Platinum|Premium|Regular)', clean, re.IGNORECASE):
            return True
        named_sections = [
            "চলুন জেনন জনই", "চলুন জেনে নেই",
            "প্ররতষ্ঠাতা ও রসইও", "প্রতিষ্ঠাতা ও সিইও",
            "কযাোশযাশেি টিকানা", "যোগাযোগের ঠিকানা",
            "আমাশদি অফিনের জলানেশন", "আমাদের অফিসের লোকেশন",
        ]
        return any(ns in clean for ns in named_sections)

    def _is_question(self, line: str) -> bool:
        clean = line.strip()
        if clean.endswith("?") or clean.endswith("? "):
            return True
        q_markers = ["েত?", "কত?", "জেমন?", "কেমন?", "কাদের জন্য?", "োনের েনয?", "আনে ফে?", "আছে কি?"]
        return any(m in clean for m in q_markers)

    def split_pages(self, pages: List[dict[str, Any]]) -> List[TextChunk]:
        flat_lines = []
        for p in pages:
            page_num = p["page"]
            source = p.get("source", "unknown.pdf")
            lines = p["text"].split("\n")
            for line in lines:
                l_s = line.strip()
                if l_s:
                    flat_lines.append({"text": l_s, "page": page_num, "source": source})

        chunks: List[TextChunk] = []
        current_section = "General Information / সাধারণ পরিচিতি"
        current_q: Optional[str] = None
        current_a_lines: List[str] = []
        q_start_page = 1
        source = pages[0].get("source", "unknown.pdf") if pages else "unknown.pdf"
        global_idx = 0

        def emit_chunk(q: Optional[str], ans_lines: List[str], page: int, section: str):
            nonlocal global_idx
            ans_text = " ".join(ans_lines).strip()
            if not q and not ans_text:
                return

            if q:
                header = f"[প্যাকেজ/সেকশন: {section}]\nপ্রশ্ন: {q}\nউত্তর: {ans_text}"
            else:
                header = f"[প্যাকেজ/সেকশন: {section}]\n{ans_text}"

            if len(header) > self.max_chunk_size and self.fallback_splitter:
                sub_chunks = self.fallback_splitter._split_text_recursively(header, self.fallback_splitter.separators)
                for sc in sub_chunks:
                    c = TextChunk(
                        text=sc,
                        chunk_index=global_idx,
                        page=page,
                        source=source,
                        char_count=len(sc),
                        estimated_tokens=max(1, len(sc) // 4),
                        section=section,
                    )
                    chunks.append(c)
                    global_idx += 1
            else:
                c = TextChunk(
                    text=header,
                    chunk_index=global_idx,
                    page=page,
                    source=source,
                    char_count=len(header),
                    estimated_tokens=max(1, len(header) // 4),
                    section=section,
                )
                chunks.append(c)
                global_idx += 1

        i = 0
        while i < len(flat_lines):
            item = flat_lines[i]
            line = item["text"]
            page = item["page"]

            if self._is_section_header(line):
                emit_chunk(current_q, current_a_lines, q_start_page, current_section)
                current_q = None
                current_a_lines = []
                current_section = line
                q_start_page = page
                i += 1
                continue

            if (line in ["হনব্?", "হবে?"] or line.endswith("?")) and current_q and not self._is_question(current_q):
                current_q = f"{current_q} {line}"
                i += 1
                continue

            if i + 1 < len(flat_lines) and (flat_lines[i + 1]["text"] in ["হনব্?", "হবে?"] or (flat_lines[i + 1]["text"].endswith("?") and len(line) < 90 and not line.endswith("।"))):
                emit_chunk(current_q, current_a_lines, q_start_page, current_section)
                current_q = f"{line} {flat_lines[i + 1]['text']}"
                current_a_lines = []
                q_start_page = page
                i += 2
                continue

            if self._is_question(line):
                emit_chunk(current_q, current_a_lines, q_start_page, current_section)
                current_q = line
                current_a_lines = []
                q_start_page = page
                i += 1
                continue

            current_a_lines.append(line)
            i += 1

        emit_chunk(current_q, current_a_lines, q_start_page, current_section)
        return chunks

