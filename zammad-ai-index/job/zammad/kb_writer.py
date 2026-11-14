"""Write access to the Zammad knowledge base, for the docs sync. Uses a token with the knowledge_base.editor permission."""

from dataclasses import dataclass
from typing import Any

from pydantic import SecretStr

from job.settings.zammad import ZammadAPISettings
from job.zammad.api import ZammadAPIClient


@dataclass(frozen=True, slots=True)
class KBCategory:
    """A category as stored in Zammad."""

    id: int
    parent_id: int | None
    title: str


@dataclass(frozen=True, slots=True)
class KBAnswer:
    """An answer as stored in Zammad."""

    id: int
    category_id: int
    title: str
    translation_id: int | None
    archived: bool
    published: bool


@dataclass(frozen=True, slots=True)
class KBTree:
    """The knowledge base's categories and answers, in the primary locale."""

    locale_id: int
    categories: dict[int, KBCategory]
    answers: dict[int, KBAnswer]

    def subtree(self, root_id: int) -> set[int]:
        """Return the root category and every category below it."""
        ids = {root_id}
        changed = True
        while changed:
            changed = False
            for category in self.categories.values():
                if category.parent_id in ids and category.id not in ids:
                    ids.add(category.id)
                    changed = True
        return ids


def _assets(data: Any, name: str) -> dict[str, dict[str, Any]]:
    # `init` returns the assets at the top level; single-object endpoints wrap them in "assets".
    assets = data.get("assets", data) if isinstance(data, dict) else {}
    found = assets.get(name, {})
    return found if isinstance(found, dict) else {}


class KBWriter:
    """Create, update, publish and archive knowledge base categories and answers."""

    def __init__(self, settings: ZammadAPISettings, editor_token: SecretStr) -> None:
        """Connect to Zammad with the KB editor token instead of the read-only index token."""
        self.kb_id = settings.knowledge_base_id
        self.client = ZammadAPIClient(settings.model_copy(update={"auth_token": editor_token}))

    def _request(self, method: str, url: str, **kwargs: Any) -> Any:
        return self.client._request(method, url, **kwargs)

    def read_tree(self) -> KBTree:
        """Read every category and answer of the knowledge base, with titles in its primary locale."""
        data = self._request("POST", "/api/v1/knowledge_bases/init")
        locales = [
            loc for loc in _assets(data, "KnowledgeBaseLocale").values() if loc.get("knowledge_base_id") == self.kb_id
        ]
        if not locales:
            raise ValueError(f"Knowledge base {self.kb_id} has no locale")
        locale = next((loc for loc in locales if loc.get("primary")), locales[0])
        locale_id = int(locale["id"])

        category_titles = {
            int(t["category_id"]): str(t.get("title") or "")
            for t in _assets(data, "KnowledgeBaseCategoryTranslation").values()
            if t.get("kb_locale_id") == locale_id
        }
        categories = {
            int(c["id"]): KBCategory(int(c["id"]), c.get("parent_id"), category_titles.get(int(c["id"]), ""))
            for c in _assets(data, "KnowledgeBaseCategory").values()
            if c.get("knowledge_base_id") == self.kb_id
        }
        answer_translations = {
            int(t["answer_id"]): t
            for t in _assets(data, "KnowledgeBaseAnswerTranslation").values()
            if t.get("kb_locale_id") == locale_id
        }
        answers: dict[int, KBAnswer] = {}
        for a in _assets(data, "KnowledgeBaseAnswer").values():
            if int(a["category_id"]) not in categories:
                continue
            translation = answer_translations.get(int(a["id"]), {})
            answers[int(a["id"])] = KBAnswer(
                id=int(a["id"]),
                category_id=int(a["category_id"]),
                title=str(translation.get("title") or ""),
                translation_id=translation.get("id"),
                archived=bool(a.get("archived_at")),
                published=bool(a.get("published_at")),
            )
        return KBTree(locale_id=locale_id, categories=categories, answers=answers)

    def create_category(self, *, parent_id: int | None, title: str, locale_id: int, icon: str = "f115") -> int:
        """Create a category and return its ID."""
        data = self._request(
            "POST",
            f"/api/v1/knowledge_bases/{self.kb_id}/categories",
            json={
                "knowledge_base_id": self.kb_id,
                "parent_id": parent_id,
                "category_icon": icon,
                "translations_attributes": [{"kb_locale_id": locale_id, "title": title}],
            },
        )
        return int(data["id"])

    def create_answer(self, *, category_id: int, title: str, body: str, locale_id: int) -> int:
        """Create an answer as a draft and return its ID."""
        data = self._request(
            "POST",
            f"/api/v1/knowledge_bases/{self.kb_id}/answers",
            json={
                "category_id": category_id,
                "translations_attributes": [
                    {"kb_locale_id": locale_id, "title": title, "content_attributes": {"body": body}}
                ],
            },
        )
        return int(data["id"])

    def update_answer(self, *, answer: KBAnswer, title: str, body: str, locale_id: int) -> None:
        """Replace an answer's title and body in the primary locale."""
        translation: dict[str, Any] = {"kb_locale_id": locale_id, "title": title, "content_attributes": {"body": body}}
        if answer.translation_id is not None:
            translation["id"] = answer.translation_id
        self._request(
            "PATCH",
            f"/api/v1/knowledge_bases/{self.kb_id}/answers/{answer.id}",
            json={"category_id": answer.category_id, "translations_attributes": [translation]},
        )

    def publish(self, answer_id: int) -> None:
        """Publish an answer, which also puts it in the KB feed the index job reads."""
        self._request("POST", f"/api/v1/knowledge_bases/{self.kb_id}/answers/{answer_id}/publish")

    def archive(self, answer_id: int) -> None:
        """Archive an answer; it stays restorable in Zammad."""
        self._request("POST", f"/api/v1/knowledge_bases/{self.kb_id}/answers/{answer_id}/archive")

    def unarchive(self, answer_id: int) -> None:
        """Bring an archived answer back."""
        self._request("POST", f"/api/v1/knowledge_bases/{self.kb_id}/answers/{answer_id}/unarchive")
