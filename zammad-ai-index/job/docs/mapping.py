"""Map a project's docs repository to the KB categories and answers the sync should maintain."""

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from job.docs.convert import _segments, is_arabic, markdown_to_html, parse_mdx
from job.projects import DocsSection, ProjectConfig


@dataclass(frozen=True, slots=True)
class PlannedCategory:
    """A KB category under the project root, identified by its title path."""

    path: tuple[str, ...]
    position: int


@dataclass(frozen=True, slots=True)
class PlannedArticle:
    """A KB answer the docs repo defines."""

    key: str
    category_path: tuple[str, ...]
    title: str
    html: str
    position: int
    source_url: str
    content_hash: str = field(init=False)

    def __post_init__(self) -> None:
        """Hash everything the sync writes, so an unchanged answer is skipped."""
        digest = hashlib.sha256("\x1f".join([*self.category_path, self.title, self.html]).encode()).hexdigest()
        object.__setattr__(self, "content_hash", digest)


@dataclass(slots=True)
class DocsPlan:
    """Everything the docs repo defines for one project."""

    categories: list[PlannedCategory] = field(default_factory=list)
    articles: list[PlannedArticle] = field(default_factory=list)

    def add_category(self, path: tuple[str, ...]) -> None:
        """Add a category and its parents, once, in first-seen order."""
        for depth in range(1, len(path) + 1):
            prefix = path[:depth]
            if all(category.path != prefix for category in self.categories):
                self.categories.append(PlannedCategory(path=prefix, position=len(self.categories)))


def page_url(site_url: str, relative: Path) -> str:
    """Return the docs-site URL of a page: content/docs/a/b.mdx -> /a/b, content/api/x/index.mdx -> /api/x."""
    parts = list(relative.with_suffix("").parts)
    if parts[:2] == ["content", "docs"]:
        parts = parts[2:]
    elif parts[:1] == ["content"]:
        parts = parts[1:]
    if parts and parts[-1] == "index":
        parts = parts[:-1]
    return site_url.rstrip("/") + "/" + "/".join(parts)


def _with_source(markdown: str, url: str, arabic: bool) -> str:
    label = "المصدر" if arabic else "Source"
    return f"{markdown}\n\n{label}: [{url}]({url})"


def _faq_articles(project: ProjectConfig, section: DocsSection, root: Path, plan: DocsPlan) -> None:
    file = root / section.path
    page = parse_mdx(file.read_text(encoding="utf-8"))
    site_url = str(project.docs.site_url) if project.docs else ""
    url = page_url(site_url, Path(section.path))
    base = (section.category,) if section.category else ()
    group: str | None = None
    question: str | None = None
    body: list[str] = []

    def flush() -> None:
        if question is None or group is None:
            return
        answer = re.sub(r"(\n\s*-{3,}\s*)+\Z", "", "".join(body).strip()).strip()
        path = (*base, group)
        plan.add_category(path)
        plan.articles.append(
            PlannedArticle(
                key=f"faq:{question}",
                category_path=path,
                title=question,
                html=markdown_to_html(_with_source(answer, url, is_arabic(question)), site_url),
                position=len(plan.articles),
                source_url=url,
            )
        )

    for code, chunk in _segments(page.markdown + "\n"):
        lines = [chunk] if code else chunk.splitlines(keepends=True)
        for line in lines:
            if not code and line.startswith("## "):
                flush()
                group, question, body = line[3:].strip(), None, []
            elif not code and line.startswith("### "):
                flush()
                question, body = line[4:].strip(), []
            elif question is not None:
                body.append(line)
    flush()


def _ordered_entries(folder: Path) -> list[Path]:
    """Return a folder's pages and subfolders in navigation order: meta.json first, then index, then by name."""
    entries = {p.stem if p.is_file() else p.name: p for p in folder.iterdir() if p.suffix == ".mdx" or p.is_dir()}
    ordered: list[Path] = []
    meta = folder / "meta.json"
    if meta.exists():
        for name in json.loads(meta.read_text(encoding="utf-8")).get("pages", []):
            if name in entries and entries[name] not in ordered:
                ordered.append(entries[name])
    rest = sorted((p for p in entries.values() if p not in ordered), key=lambda p: (p.stem != "index", p.name))
    return ordered + rest


def _folder_title(folder: Path) -> str:
    meta = folder / "meta.json"
    if meta.exists():
        title = json.loads(meta.read_text(encoding="utf-8")).get("title")
        if title:
            return str(title)
    index = folder / "index.mdx"
    if index.exists():
        title = parse_mdx(index.read_text(encoding="utf-8")).title
        if title:
            return title
    return folder.name


def _guide_articles(
    project: ProjectConfig, section: DocsSection, root: Path, plan: DocsPlan, claimed: set[Path]
) -> None:
    site_url = str(project.docs.site_url) if project.docs else ""
    base = (section.category,) if section.category else ()

    def walk(folder: Path, category: tuple[str, ...]) -> None:
        if category:
            plan.add_category(category)
        for entry in _ordered_entries(folder):
            if entry.is_dir():
                walk(entry, (*category, _folder_title(entry)))
                continue
            if entry.resolve() in claimed:
                continue
            relative = entry.relative_to(root)
            page = parse_mdx(entry.read_text(encoding="utf-8"))
            url = page_url(site_url, relative)
            title = page.title or entry.stem
            markdown = f"{page.description}\n\n{page.markdown}" if page.description else page.markdown
            plan.articles.append(
                PlannedArticle(
                    key=f"page:{relative.with_suffix('').as_posix()}",
                    category_path=category,
                    title=title,
                    html=markdown_to_html(_with_source(markdown, url, is_arabic(title)), site_url),
                    position=len(plan.articles),
                    source_url=url,
                )
            )

    walk(root / section.path, base)


def build_plan(project: ProjectConfig, root: Path) -> DocsPlan:
    """Build the KB categories and answers a project's docs repo (checked out at root) defines."""
    if project.docs is None:
        raise ValueError(f"Project {project.key} has no docs repository")
    plan = DocsPlan()
    claimed = {(root / s.path).resolve() for s in project.docs.sections if s.path.endswith(".mdx")}
    for section in project.docs.sections:
        if section.kind == "faq":
            _faq_articles(project, section, root, plan)
        else:
            _guide_articles(project, section, root, plan, claimed)
    keys = [article.key for article in plan.articles]
    duplicates = sorted({key for key in keys if keys.count(key) > 1})
    if duplicates:
        raise ValueError(f"Docs define the same answer twice: {', '.join(duplicates)}")
    return plan
