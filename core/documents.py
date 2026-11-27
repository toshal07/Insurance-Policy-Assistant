"""Page-preserving, clause-aware PDF extraction and LangChain chunking."""
import base64
import io
import re
import uuid
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

_KEYWORD_CLAUSE_RE = re.compile(
    r'^\s*(?:clause|section|article|condition|part|schedule)\s+([A-Za-z0-9]+(?:\.[A-Za-z0-9]+)*)'
    r'(?:\s*[:.\-–—]\s*(.+)|\s+([A-Z][^\n]*))?$',
    re.IGNORECASE,
)
_NUMBERED_SUBCLAUSE_RE = re.compile(r'^\s*(\d+(?:\.\d+)+)\.?\s+([A-Z][^\n]{1,90})$')
_TOP_NUMBERED_CLAUSE_RE = re.compile(r'^\s*(\d+)\.\s+([A-Z][A-Za-z0-9\s,&/()\'’-]{2,70})$')

_TEXT_SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=1200,
    chunk_overlap=200,
    separators=['\n\n', '\n', '. ', '; ', ' ', ''],
)


def _clean_title(raw_title: str | None) -> str | None:
    if not raw_title:
        return None
    title = re.sub(r'\s+', ' ', raw_title).strip(' :.-–—')
    if not title or len(title) > 80 or len(title.split()) > 10:
        return None
    return title


def _detect_clause_header(line: str) -> tuple[str, str | None] | None:
    stripped = line.strip()
    if not stripped or len(stripped) > 120:
        return None
    match = _KEYWORD_CLAUSE_RE.match(stripped)
    if match:
        return match.group(1), _clean_title(match.group(2) or match.group(3))
    match = _NUMBERED_SUBCLAUSE_RE.match(stripped)
    if match:
        title = _clean_title(match.group(2))
        if title:
            return match.group(1), title
    match = _TOP_NUMBERED_CLAUSE_RE.match(stripped)
    if match and not (stripped.endswith('.') and len(stripped.split()) > 5):
        title = _clean_title(match.group(2))
        if title:
            return match.group(1), title
    return None


def _split_page_by_clause(
    text: str,
    initial_clause: str | None,
    initial_title: str | None,
) -> tuple[list[tuple[str | None, str | None, str]], str | None, str | None]:
    segments: list[tuple[str | None, str | None, str]] = []
    current_clause = initial_clause
    current_title = initial_title
    buffer: list[str] = []

    for line in text.splitlines():
        detected = _detect_clause_header(line)
        if detected:
            body = '\n'.join(buffer).strip()
            if body:
                segments.append((current_clause, current_title, body))
                buffer = []
            current_clause, current_title = detected
        buffer.append(line)

    tail = '\n'.join(buffer).strip()
    if tail:
        segments.append((current_clause, current_title, tail))
    return segments, current_clause, current_title


def to_langchain_documents(doc: dict) -> list[Document]:
    """Convert extracted document chunks into LangChain Document objects."""
    return [
        Document(
            page_content=chunk['text'],
            metadata={
                'page': chunk['page'],
                'clause': chunk.get('clause'),
                'clause_title': chunk.get('clause_title'),
                'chunk_index': chunk.get('chunk_index', idx),
                'document': doc['name'],
                'document_id': doc['id'],
                'category': doc.get('category', 'Insurance'),
            },
        )
        for idx, chunk in enumerate(doc.get('chunks', []))
    ]


def extract_document(payload: dict, limit_mb: int) -> dict:
    """Validate an uploaded PDF payload and extract page- and clause-linked chunks."""
    raw = base64.b64decode(payload['data'], validate=True)
    if len(raw) > limit_mb * 1024 * 1024:
        raise ValueError(f'Please choose a PDF smaller than {limit_mb} MB.')
    if not raw.startswith(b'%PDF-'):
        raise ValueError('This file is not a valid PDF.')
    try:
        pdf = PdfReader(io.BytesIO(raw))
        if pdf.is_encrypted:
            raise ValueError('This PDF is password protected. Upload an unlocked copy.')
        pages = [{'page': i + 1, 'text': p.extract_text() or ''} for i, p in enumerate(pdf.pages)]
    except ValueError:
        raise
    except Exception:
        raise ValueError('This PDF could not be read. It may be damaged or unsupported.') from None
    if not any(p['text'].strip() for p in pages):
        raise ValueError('No readable text found. Scan-only PDFs need OCR before upload.')

    chunks = []
    active_clause: str | None = None
    active_title: str | None = None
    for page in pages:
        text = page['text'].strip()
        if not text:
            continue
        segments, active_clause, active_title = _split_page_by_clause(text, active_clause, active_title)
        for clause, clause_title, segment_text in segments:
            for excerpt in _TEXT_SPLITTER.split_text(segment_text):
                cleaned = excerpt.strip()
                if cleaned:
                    chunks.append({
                        'page': page['page'],
                        'text': cleaned,
                        'clause': clause,
                        'clause_title': clause_title,
                        'chunk_index': len(chunks),
                    })

    return {
        'id': uuid.uuid4().hex,
        'name': payload['name'][:200],
        'category': payload.get('category', 'Insurance'),
        'pages': len(pages),
        'size': len(raw),
        'chunks': chunks,
    }
