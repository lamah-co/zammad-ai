"""Projects: each answers from its own Qdrant collection, selected by the event's request type (the project key).

Project files are shared with the index job and the deploy (`docker/zammad-ai/projects/<key>.yaml` in zammad-infra).
The workflow only needs the key and the collection; the rest of each file is ignored here.
"""

from pathlib import Path

import yaml
from pydantic import BaseModel, Field

from app.action.service import ActionService
from app.answer.service import AnswerService
from app.settings import ZammadAISettings
from app.utils.logging import getLogger


class _ProjectQdrant(BaseModel):
    collection_name: str


class ProjectUseCase(BaseModel):
    """The part of a project file the workflow uses."""

    key: str = Field(pattern=r"^[a-z][a-z0-9_-]*$")
    qdrant: _ProjectQdrant


def load_project_use_cases(directory: Path) -> dict[str, ProjectUseCase]:
    """Load every `*.yaml` project file, keyed by project key.

    Raises:
        ValueError: If two projects share a key or a collection, or a file is not named after its key.
    """
    projects: dict[str, ProjectUseCase] = {}
    for path in sorted(directory.glob("*.yaml")):
        project = ProjectUseCase.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
        if project.key != path.stem:
            raise ValueError(f"Project file {path.name} must be named after its key {project.key!r}")
        projects[project.key] = project
    collections = [p.qdrant.collection_name for p in projects.values()]
    if len(set(collections)) != len(collections):
        raise ValueError("Two projects share a Qdrant collection")
    return projects


def project_settings(settings: ZammadAISettings, project: ProjectUseCase) -> ZammadAISettings:
    """Return settings that answer from the project's collection, filtered to the project's vectors."""
    qdrant = settings.answer.qdrant.model_copy(
        update={"collection_name": project.qdrant.collection_name, "project": project.key}
    )
    answer = settings.answer.model_copy(update={"qdrant": qdrant})
    return settings.model_copy(update={"answer": answer})


def build_project_actions(settings: ZammadAISettings) -> dict[str, ActionService]:
    """Build one answer and action service per project; empty when no projects directory is configured."""
    if settings.projects_dir is None:
        return {}
    actions: dict[str, ActionService] = {}
    for key, project in load_project_use_cases(settings.projects_dir).items():
        scoped = project_settings(settings, project)
        actions[key] = ActionService(settings=scoped, answer_service=AnswerService(settings=scoped))
    getLogger("zammad-ai.projects").info(
        f"Answering {len(actions)} project(s) from their own collections: {', '.join(actions) or 'none'}"
    )
    return actions
