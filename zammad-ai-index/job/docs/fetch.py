"""Download a docs repository as a tarball through the GitHub API, so the job needs no git or SSH key."""

import tarfile
import tempfile
import urllib.request
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from pydantic import SecretStr

GITHUB_API = "https://api.github.com"


@contextmanager
def checkout(repo: str, branch: str, token: SecretStr, *, api_url: str = GITHUB_API) -> Iterator[tuple[Path, str]]:
    """Yield the extracted repository root and its commit SHA; the files are removed afterwards.

    Raises:
        ValueError: If the archive does not hold exactly one top-level folder.
    """
    request = urllib.request.Request(
        f"{api_url}/repos/{repo}/tarball/{branch}",
        headers={
            "Authorization": f"Bearer {token.get_secret_value()}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "zammad-ai-index",
        },
    )
    with tempfile.TemporaryDirectory(prefix="docs-") as work:
        archive = Path(work) / "repo.tar.gz"
        with urllib.request.urlopen(request, timeout=120) as response, archive.open("wb") as out:  # noqa: S310
            while chunk := response.read(1 << 20):
                out.write(chunk)
        target = Path(work) / "src"
        with tarfile.open(archive) as tar:
            tar.extractall(target, filter="data")
        roots = [p for p in target.iterdir() if p.is_dir()]
        if len(roots) != 1:
            raise ValueError(f"Unexpected archive layout for {repo}")
        # GitHub names the folder <owner>-<repo>-<sha>.
        yield roots[0], roots[0].name.rsplit("-", 1)[-1]
