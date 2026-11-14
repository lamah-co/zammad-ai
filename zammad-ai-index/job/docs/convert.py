"""Convert Fumadocs MDX pages into the HTML Zammad stores for knowledge base answers."""

import logging
import re
import textwrap
from dataclasses import dataclass
from typing import Any

import yaml
from markdown_it import MarkdownIt

_FENCE = re.compile(r"^(```|~~~)")
_FRONTMATTER = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.DOTALL)
_ARABIC = re.compile(r"[؀-ۿ]")

_markdown = MarkdownIt("commonmark", {"html": False}).enable("table")
# markdown-it logs every parsing step at DEBUG; keep the job log readable.
logging.getLogger("markdown_it").setLevel(logging.WARNING)


@dataclass(frozen=True, slots=True)
class MdxPage:
    """A parsed MDX page: frontmatter and Markdown body with the JSX components resolved."""

    title: str
    description: str
    markdown: str


def is_arabic(text: str) -> bool:
    """Return whether the text contains Arabic script."""
    return bool(_ARABIC.search(text))


def split_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    """Return the YAML frontmatter and the rest of the page."""
    match = _FRONTMATTER.match(text)
    if not match:
        return {}, text
    data = yaml.safe_load(match.group(1)) or {}
    return (data if isinstance(data, dict) else {}), text[match.end() :]


def _segments(text: str) -> list[tuple[bool, str]]:
    """Split text into (is_code, chunk) parts, so component rewriting never touches code blocks."""
    parts: list[tuple[bool, str]] = []
    buffer: list[str] = []
    in_code = False
    for line in text.splitlines(keepends=True):
        if _FENCE.match(line.lstrip()):
            if not in_code:
                if buffer:
                    parts.append((False, "".join(buffer)))
                buffer = [line]
                in_code = True
            else:
                buffer.append(line)
                parts.append((True, "".join(buffer)))
                buffer = []
                in_code = False
            continue
        buffer.append(line)
    if buffer:
        parts.append((in_code, "".join(buffer)))
    return parts


def _attr(tag: str, name: str) -> str:
    match = re.search(rf'{name}="([^"]*)"', tag) or re.search(rf"{name}='([^']*)'", tag)
    return match.group(1) if match else ""


def _rewrite_prose(text: str, arabic: bool) -> str:
    """Resolve the Fumadocs components in text outside code blocks."""
    text = "\n".join(line for line in text.split("\n") if not line.startswith("import "))

    def step_list(match: re.Match[str]) -> str:
        steps = re.findall(r"<Step>(.*?)</Step>", match.group(1), re.DOTALL)
        items: list[str] = []
        for number, body in enumerate(steps, start=1):
            lines = textwrap.dedent(body).strip().split("\n")
            heading = lines[0].lstrip("#").strip() if lines and lines[0].lstrip().startswith("#") else ""
            rest = "\n".join(lines[1:] if heading else lines).strip()
            items.append(f"**{number}. {heading}**\n\n{rest}" if heading else f"**{number}.** {rest}")
        return "\n\n".join(items)

    text = re.sub(r"<Steps>(.*?)</Steps>", step_list, text, flags=re.DOTALL)

    labels = {"info": "ملاحظة" if arabic else "Note", "warn": "تنبيه" if arabic else "Warning"}

    def callout(match: re.Match[str]) -> str:
        kind = _attr(match.group(1), "type") or "info"
        label = labels["warn" if kind.startswith("warn") else "info"]
        body = textwrap.dedent(match.group(2)).strip()
        quoted = "\n".join(f"> {line}" if line else ">" for line in body.split("\n"))
        return f"> **{label}:**\n>\n{quoted}"

    text = re.sub(r"<Callout([^>]*)>(.*?)</Callout>", callout, text, flags=re.DOTALL)

    def card(match: re.Match[str]) -> str:
        tag = match.group(0)
        title, href, description = _attr(tag, "title"), _attr(tag, "href"), _attr(tag, "description")
        line = f"- [{title}]({href})" if href else f"- {title}"
        return f"{line} — {description}" if description else line

    text = re.sub(r"<Card\b[^>]*/>", card, text)
    text = re.sub(r"</?Cards>", "", text)

    text = re.sub(r"<Tab\s+value=[\"']([^\"']*)[\"']\s*>", lambda m: f"**{m.group(1)}**\n", text)
    text = re.sub(r"</Tab>|<Tabs\b[^>]*>|</Tabs>", "", text)

    # Zammad strips external images from answers, so link the screenshot under its description instead.
    def image(match: re.Match[str]) -> str:
        tag = match.group(0)
        alt = _attr(tag, "alt") or ("صورة توضيحية" if arabic else "Screenshot")
        return f"[{alt}]({_attr(tag, 'src')})"

    return re.sub(r"<img\b[^>]*/?>", image, text)


def parse_mdx(text: str) -> MdxPage:
    """Parse an MDX page into its title, description and plain Markdown body."""
    frontmatter, body = split_frontmatter(text)
    title = str(frontmatter.get("title") or "").strip()
    arabic = is_arabic(title) or is_arabic(body)
    markdown = "".join(chunk if code else _rewrite_prose(chunk, arabic) for code, chunk in _segments(body))
    markdown = re.sub(r"\n{3,}", "\n\n", markdown).strip()
    return MdxPage(title=title, description=str(frontmatter.get("description") or "").strip(), markdown=markdown)


def markdown_to_html(markdown: str, site_url: str) -> str:
    """Render Markdown to HTML and point site-relative links and images at the docs site."""
    html = _markdown.render(markdown)
    base = site_url.rstrip("/")
    return re.sub(r'(href|src)="/(?!/)', lambda m: f'{m.group(1)}="{base}/', html)
