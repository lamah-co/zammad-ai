"""Bring a project's KB category tree in line with its docs: create, update, publish and archive answers.

An answer is identified by its category path and title, read from Zammad on every run. A small state file only
remembers content hashes, to skip unchanged answers; if it is lost, the next run updates everything once.
"""

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from job.docs.mapping import DocsPlan
from job.zammad.kb_writer import KBAnswer, KBTree


class KBWriterLike(Protocol):
    """The knowledge base operations the sync needs."""

    def read_tree(self) -> KBTree: ...  # noqa: D102
    def create_category(self, *, parent_id: int | None, title: str, locale_id: int) -> int: ...  # noqa: D102
    def create_answer(self, *, category_id: int, title: str, body: str, locale_id: int) -> int: ...  # noqa: D102
    def update_answer(self, *, answer: KBAnswer, title: str, body: str, locale_id: int) -> None: ...  # noqa: D102
    def publish(self, answer_id: int) -> None: ...  # noqa: D102
    def archive(self, answer_id: int) -> None: ...  # noqa: D102
    def unarchive(self, answer_id: int) -> None: ...  # noqa: D102


@dataclass(slots=True)
class SyncResult:
    """What a sync did, or would do in a dry run."""

    actions: list[tuple[str, str]] = field(default_factory=list)
    root_category_id: int | None = None

    def add(self, action: str, subject: str) -> None:
        """Record one action, such as ("create", "ERSAAL › FAQ › question")."""
        self.actions.append((action, subject))

    @property
    def counts(self) -> Counter[str]:
        """Return how many actions of each kind there were."""
        return Counter(action for action, _ in self.actions)

    @property
    def changed(self) -> bool:
        """Return whether the KB changed, so the project needs re-indexing."""
        return any(action != "unchanged" for action, _ in self.actions)


def _load_state(path: Path) -> dict[str, str]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError, json.JSONDecodeError:
        return {}
    return {str(k): str(v) for k, v in data.get("hashes", {}).items()} if isinstance(data, dict) else {}


def _save_state(path: Path, hashes: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"hashes": hashes}, ensure_ascii=False, indent=1), encoding="utf-8")


def sync_docs(
    plan: DocsPlan, *, root_title: str, writer: KBWriterLike, state_file: Path, dry_run: bool = False
) -> SyncResult:
    """Apply the plan below the project's root category. Nothing outside that tree is ever changed."""
    result = SyncResult()
    tree = writer.read_tree()
    locale = tree.locale_id
    hashes = _load_state(state_file)

    def child_id(parent_id: int | None, title: str) -> int | None:
        return next(
            (c.id for c in tree.categories.values() if c.parent_id == parent_id and c.title == title),
            None,
        )

    root_id = child_id(None, root_title)
    if root_id is None:
        result.add("create category", root_title)
        root_id = None if dry_run else writer.create_category(parent_id=None, title=root_title, locale_id=locale)
    result.root_category_id = root_id

    category_ids: dict[tuple[str, ...], int | None] = {(): root_id}
    for category in plan.categories:
        parent = category_ids[category.path[:-1]]
        existing = child_id(parent, category.path[-1]) if parent is not None else None
        if existing is None:
            result.add("create category", " › ".join((root_title, *category.path)))
            if not dry_run and parent is not None:
                existing = writer.create_category(parent_id=parent, title=category.path[-1], locale_id=locale)
        category_ids[category.path] = existing

    managed = tree.subtree(root_id) if root_id is not None else set()
    managed_answers = [a for a in tree.answers.values() if a.category_id in managed]
    matched: set[int] = set()
    new_hashes: dict[str, str] = {}

    for article in plan.articles:
        label = " › ".join((root_title, *article.category_path, article.title))
        category_id = category_ids[article.category_path]
        candidates = [a for a in managed_answers if a.category_id == category_id and a.title == article.title]
        answer = next((a for a in candidates if not a.archived), candidates[0] if candidates else None)
        if answer is None:
            result.add("create", label)
            if not dry_run and category_id is not None:
                answer_id = writer.create_answer(
                    category_id=category_id, title=article.title, body=article.html, locale_id=locale
                )
                writer.publish(answer_id)
                new_hashes[str(answer_id)] = article.content_hash
            continue
        matched.add(answer.id)
        if hashes.get(str(answer.id)) == article.content_hash and answer.published and not answer.archived:
            result.add("unchanged", label)
            new_hashes[str(answer.id)] = article.content_hash
            continue
        result.add("update", label)
        if not dry_run:
            if answer.archived:
                writer.unarchive(answer.id)
            writer.update_answer(answer=answer, title=article.title, body=article.html, locale_id=locale)
            if not answer.published or answer.archived:
                writer.publish(answer.id)
            new_hashes[str(answer.id)] = article.content_hash

    for answer in managed_answers:
        if answer.id not in matched and not answer.archived:
            result.add("archive", answer.title)
            if not dry_run:
                writer.archive(answer.id)

    if not dry_run:
        _save_state(state_file, new_hashes)
    return result


def archive_category(category_id: int, *, writer: KBWriterLike, dry_run: bool = False) -> SyncResult:
    """Archive every answer in one category (and below), for retiring hand-written answers the docs replace."""
    result = SyncResult()
    tree = writer.read_tree()
    if category_id not in tree.categories:
        raise ValueError(f"Category {category_id} does not exist in this knowledge base")
    for answer in tree.answers.values():
        if answer.category_id in tree.subtree(category_id) and not answer.archived:
            result.add("archive", answer.title)
            if not dry_run:
                writer.archive(answer.id)
    return result
