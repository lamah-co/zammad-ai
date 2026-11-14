"""Project configuration: one YAML file per project, shared by the sync, the index job, the workflow and the deploy."""

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, HttpUrl

from job.utils.logging import getLogger


class ProjectZammad(BaseModel):
    """Where a project's tickets arrive in Zammad."""

    group: str = Field(description="Zammad group the project's channels deliver into.")
    adopt_groups: list[str] = Field(
        default_factory=list, description="Earlier group names the deploy renames to `group` instead of creating it."
    )


class ProjectKB(BaseModel):
    """The project's part of the Zammad knowledge base."""

    root_category: str = Field(description="Title of the top-level KB category the sync owns.")


class ProjectQdrant(BaseModel):
    """The project's vector collection."""

    collection_name: str = Field(description="Qdrant collection holding only this project's answers.")


class DocsSection(BaseModel):
    """A part of the docs repo and how it maps to KB answers."""

    path: str = Field(description="File or folder in the repo, for example content/docs or content/docs/faq.mdx.")
    kind: Literal["faq", "guide"] = Field(description="faq: one answer per ### question; guide: one answer per page.")
    category: str | None = Field(default=None, description="Category under the project root; none puts it at the root.")


class ProjectDocs(BaseModel):
    """The project's docs repository."""

    repo: str = Field(description="GitHub repository, owner/name.")
    branch: str = Field(default="main", description="Branch to sync from.")
    site_url: HttpUrl = Field(description="Public docs site; site-relative links and images point here.")
    sections: list[DocsSection] = Field(min_length=1, description="Sections to sync, in order.")


class ProjectConfig(BaseModel):
    """One project served by the shared Zammad."""

    key: str = Field(pattern=r"^[a-z][a-z0-9_-]*$", description="Project key; also the request type (anliegenart).")
    zammad: ProjectZammad
    kb: ProjectKB
    qdrant: ProjectQdrant
    docs: ProjectDocs | None = Field(default=None, description="Docs repo to sync from; none means a manual KB tree.")


def load_projects(directory: Path) -> dict[str, ProjectConfig]:
    """Load every `*.yaml` project file in the directory, keyed by project key.

    Raises:
        ValueError: If two projects share a key, a group, a KB root category or a Qdrant collection.
    """
    projects: dict[str, ProjectConfig] = {}
    for path in sorted(directory.glob("*.yaml")):
        project = ProjectConfig.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
        if project.key != path.stem:
            raise ValueError(f"Project file {path.name} must be named after its key {project.key!r}")
        projects[project.key] = project
    for label, values in (
        ("group", [p.zammad.group for p in projects.values()]),
        ("KB root category", [p.kb.root_category for p in projects.values()]),
        ("Qdrant collection", [p.qdrant.collection_name for p in projects.values()]),
    ):
        duplicates = sorted({value for value in values if values.count(value) > 1})
        if duplicates:
            raise ValueError(f"Projects share a {label}: {', '.join(duplicates)}")
    getLogger("zammad-ai-index.projects").info(f"Loaded {len(projects)} project(s): {', '.join(projects) or 'none'}")
    return projects
