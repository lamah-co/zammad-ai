"""Tests for syncing a docs plan into the KB: create, update, skip, archive, and never touch other categories."""

import io
import tarfile
from pathlib import Path

import pytest
from pydantic import SecretStr

from job.docs import fetch
from job.docs.mapping import DocsPlan, PlannedArticle
from job.docs.sync import archive_category, sync_docs
from job.zammad.kb_writer import KBAnswer, KBCategory, KBTree

LOCALE = 1


class FakeKB:
    """An in-memory knowledge base that records every write."""

    def __init__(self) -> None:
        """Start with one hand-written category outside any project."""
        self.categories: dict[int, KBCategory] = {2: KBCategory(2, None, "أسئلة عامة")}
        self.answers: dict[int, KBAnswer] = {50: KBAnswer(50, 2, "Manual answer", 500, archived=False, published=True)}
        self.writes: list[tuple[str, int | str]] = []
        self._next = 100

    def _id(self) -> int:
        self._next += 1
        return self._next

    def read_tree(self) -> KBTree:
        """Return the current state."""
        return KBTree(locale_id=LOCALE, categories=dict(self.categories), answers=dict(self.answers))

    def create_category(self, *, parent_id: int | None, title: str, locale_id: int) -> int:
        """Create a category."""
        new_id = self._id()
        self.categories[new_id] = KBCategory(new_id, parent_id, title)
        self.writes.append(("create_category", title))
        return new_id

    def create_answer(self, *, category_id: int, title: str, body: str, locale_id: int) -> int:
        """Create a draft answer."""
        new_id = self._id()
        self.answers[new_id] = KBAnswer(new_id, category_id, title, new_id * 10, archived=False, published=False)
        self.writes.append(("create_answer", title))
        return new_id

    def update_answer(self, *, answer: KBAnswer, title: str, body: str, locale_id: int) -> None:
        """Update an answer."""
        self.writes.append(("update_answer", answer.id))

    def publish(self, answer_id: int) -> None:
        """Publish an answer."""
        a = self.answers[answer_id]
        self.answers[answer_id] = KBAnswer(
            a.id, a.category_id, a.title, a.translation_id, archived=False, published=True
        )
        self.writes.append(("publish", answer_id))

    def archive(self, answer_id: int) -> None:
        """Archive an answer."""
        a = self.answers[answer_id]
        self.answers[answer_id] = KBAnswer(
            a.id, a.category_id, a.title, a.translation_id, archived=True, published=True
        )
        self.writes.append(("archive", answer_id))

    def unarchive(self, answer_id: int) -> None:
        """Bring an answer back."""
        self.writes.append(("unarchive", answer_id))


def plan(*articles: tuple[tuple[str, ...], str, str]) -> DocsPlan:
    """Build a plan from (category path, title, html) triples."""
    result = DocsPlan()
    for path, title, html in articles:
        result.add_category(path)
        result.articles.append(PlannedArticle(f"page:{title}", path, title, html, len(result.articles), "u"))
    return result


DOCS = plan(
    (("FAQ", "General"), "What is OTP?", "<p>One-time code.</p>"),
    (("Senders",), "Request a sender", "<p>Steps.</p>"),
)


def test_first_sync_creates_and_publishes_the_tree(tmp_path: Path) -> None:
    """An empty project gets its root, categories and published answers."""
    kb = FakeKB()

    result = sync_docs(DOCS, root_title="ERSAAL", writer=kb, state_file=tmp_path / "s.json")

    assert result.counts["create"] == 2
    assert ("create_category", "ERSAAL") in kb.writes
    created = [a for a in kb.answers.values() if a.title in {"What is OTP?", "Request a sender"}]
    assert all(a.published for a in created)


def test_second_sync_without_changes_writes_nothing(tmp_path: Path) -> None:
    """A re-run on the same docs is a no-op, so it never needs re-indexing."""
    kb = FakeKB()
    sync_docs(DOCS, root_title="ERSAAL", writer=kb, state_file=tmp_path / "s.json")
    kb.writes.clear()

    result = sync_docs(DOCS, root_title="ERSAAL", writer=kb, state_file=tmp_path / "s.json")

    assert kb.writes == []
    assert not result.changed


def test_edited_page_is_updated_and_removed_page_archived(tmp_path: Path) -> None:
    """One edit and one removal give exactly one update and one archive."""
    kb = FakeKB()
    sync_docs(DOCS, root_title="ERSAAL", writer=kb, state_file=tmp_path / "s.json")
    kb.writes.clear()
    edited = plan((("FAQ", "General"), "What is OTP?", "<p>A one-time code, valid 5 minutes.</p>"))

    result = sync_docs(edited, root_title="ERSAAL", writer=kb, state_file=tmp_path / "s.json")

    assert result.counts["update"] == 1 and result.counts["archive"] == 1
    assert [w[0] for w in kb.writes] == ["update_answer", "archive"]


def test_answers_outside_the_project_tree_are_never_touched(tmp_path: Path) -> None:
    """The hand-written answer in another category survives every sync."""
    kb = FakeKB()

    sync_docs(DOCS, root_title="ERSAAL", writer=kb, state_file=tmp_path / "s.json")
    sync_docs(plan(), root_title="ERSAAL", writer=kb, state_file=tmp_path / "s.json")

    assert not kb.answers[50].archived
    assert all(w[1] != 50 for w in kb.writes)


def test_dry_run_reports_the_plan_and_writes_nothing(tmp_path: Path) -> None:
    """A dry run on an empty KB lists the root, categories and answers, without writes or a state file."""
    kb = FakeKB()

    result = sync_docs(DOCS, root_title="ERSAAL", writer=kb, state_file=tmp_path / "s.json", dry_run=True)

    assert kb.writes == []
    assert result.counts["create"] == 2 and result.counts["create category"] == 4
    assert not (tmp_path / "s.json").exists()


def test_lost_state_updates_once_then_settles(tmp_path: Path) -> None:
    """Without the state file every answer is updated once; the next run is a no-op again."""
    kb = FakeKB()
    sync_docs(DOCS, root_title="ERSAAL", writer=kb, state_file=tmp_path / "s.json")
    (tmp_path / "s.json").unlink()

    first = sync_docs(DOCS, root_title="ERSAAL", writer=kb, state_file=tmp_path / "s.json")
    second = sync_docs(DOCS, root_title="ERSAAL", writer=kb, state_file=tmp_path / "s.json")

    assert first.counts["update"] == 2
    assert not second.changed


def test_archive_category_retires_hand_written_answers() -> None:
    """The one-off retirement of the manual FAQ archives exactly that category's answers."""
    kb = FakeKB()

    result = archive_category(2, writer=kb)

    assert result.counts["archive"] == 1 and kb.answers[50].archived
    with pytest.raises(ValueError, match="does not exist"):
        archive_category(999, writer=kb)


def test_checkout_extracts_the_tarball_and_reads_the_commit(monkeypatch: pytest.MonkeyPatch) -> None:
    """The GitHub tarball is unpacked safely; the commit SHA comes from its folder name."""
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        data = b"---\ntitle: Hi\n---\n"
        info = tarfile.TarInfo("lamah-co-ersaal-docs-abc1234/content/docs/index.mdx")
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))
    requests: list[str] = []

    def fake_urlopen(request, timeout: int):  # noqa: ANN001
        requests.append(request.full_url)
        assert request.get_header("Authorization") == "Bearer t"
        return io.BytesIO(buffer.getvalue())

    monkeypatch.setattr(fetch.urllib.request, "urlopen", fake_urlopen)

    with fetch.checkout("lamah-co/ersaal-docs", "main", SecretStr("t")) as (root, sha):
        assert (root / "content" / "docs" / "index.mdx").read_text(encoding="utf-8").startswith("---")
        assert sha == "abc1234"
    assert requests == ["https://api.github.com/repos/lamah-co/ersaal-docs/tarball/main"]


def test_full_indexing_of_a_category_tree_uses_the_category_walk(monkeypatch: pytest.MonkeyPatch) -> None:
    """The KB record's cached answer list can be stale, so a category run never relies on it."""
    from unittest.mock import MagicMock

    from job.data import retrieval

    monkeypatch.setattr(retrieval.settings.index, "full_indexing", True)
    client = MagicMock()

    answer_ids = retrieval.retrieve_answer_ids(client, [33, 31, 32])

    assert answer_ids == [31, 32, 33]
    client.kb_info.assert_not_called()
