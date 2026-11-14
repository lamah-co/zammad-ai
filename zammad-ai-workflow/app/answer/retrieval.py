"""Record the documents retrieved while answering one question, so the judge can check the answer against them."""

from collections.abc import Iterable, Iterator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar

from langchain_core.documents import Document

_retrieved: ContextVar[list[Document] | None] = ContextVar("retrieved_documents", default=None)


@contextmanager
def collect_retrieved_documents() -> Iterator[list[Document]]:
    """Collect every document retrieved inside the block, including from tools and child tasks."""
    documents: list[Document] = []
    token = _retrieved.set(documents)
    try:
        yield documents
    finally:
        _retrieved.reset(token)


def record_retrieved_documents(documents: Iterable[Document]) -> None:
    """Add retrieved documents to the active collection; outside one, do nothing."""
    collected = _retrieved.get()
    if collected is not None:
        collected.extend(documents)


def document_title(document: Document) -> str:
    """Return a document's title from its metadata."""
    return str(document.metadata.get("title") or document.metadata.get("name") or "Untitled")


def document_url(document: Document) -> str:
    """Return a document's URL from its metadata."""
    metadata = document.metadata
    return str(metadata.get("url") or metadata.get("source_url") or metadata.get("source") or "")


def judge_sources(documents: Sequence[Document], *, max_documents: int, max_chars: int) -> str:
    """Render the retrieved documents for the judge: unique, in retrieval order, each truncated to max_chars."""
    seen: set[tuple[str, str]] = set()
    parts: list[str] = []
    for document in documents:
        key = (document_url(document) or document_title(document), document.page_content)
        if key in seen:
            continue
        seen.add(key)
        content = document.page_content
        if len(content) > max_chars:
            content = content[:max_chars] + " […]"
        parts.append(
            f"[{len(parts) + 1}] {document_title(document)}\nURL: {document_url(document)}\nContent:\n{content}"
        )
        if len(parts) >= max_documents:
            break
    return "\n\n".join(parts) if parts else "No documents were retrieved."
