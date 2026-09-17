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

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "chunk_index": self.chunk_index,
            "page": self.page,
            "source": self.source,
            "char_count": self.char_count,
            "estimated_tokens": self.estimated_tokens,
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
