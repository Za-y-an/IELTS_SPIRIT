from pathlib import Path
from typing import Any, List
import httpx
from pypdf import PdfReader


async def download_pdf(url: str, destination_dir: Path, timeout: float = 30.0) -> Path:
    """
    Download a PDF file asynchronously from the given URL and store it in destination_dir.
    Returns the resolved Path of the downloaded file.
    """
    destination_dir.mkdir(parents=True, exist_ok=True)

    # Derive filename from URL path or fallback to a standard name
    url_filename = Path(url.split("?")[0]).name
    if not url_filename.lower().endswith(".pdf"):
        url_filename = "document.pdf"
    
    file_path = destination_dir / url_filename

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }

    async with httpx.AsyncClient(headers=headers, follow_redirects=True, timeout=timeout) as client:
        response = await client.get(url)
        response.raise_for_status()
        
        # Write bytes asynchronously
        file_path.write_bytes(response.content)

    return file_path


def extract_text_from_pdf(pdf_path: Path) -> List[dict[str, Any]]:
    """
    Extract text page by page from the PDF file using pypdf.
    Returns a list of page dicts: [{'page': int, 'text': str, 'source': str}, ...]
    """
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found at: {pdf_path}")

    reader = PdfReader(str(pdf_path))
    pages_data: List[dict[str, Any]] = []

    for idx, page in enumerate(reader.pages, start=1):
        extracted = page.extract_text() or ""
        cleaned = extracted.strip()
        if cleaned:
            pages_data.append({
                "page": idx,
                "text": cleaned,
                "source": pdf_path.name,
            })

    return pages_data
