"""Tests for converting a project's MDX docs into KB categories and answers, on sample Fumadocs pages."""

from pathlib import Path

import pytest

from job.docs.convert import markdown_to_html, parse_mdx
from job.docs.mapping import build_plan, page_url
from job.projects import ProjectConfig, load_projects

# Made-up pages with the structure of a Fumadocs docs repo (FAQ, guides with Steps/Callout/Cards/images, API pages).
FIXTURES = Path(__file__).parent / "fixtures" / "sample"
SITE = "https://docs.getersaal.com"


def ersaal(sections: list[dict]) -> ProjectConfig:
    """Return the ERSAAL project config with the given docs sections."""
    return ProjectConfig.model_validate(
        {
            "key": "ersaal",
            "zammad": {"group": "ERSAAL Support"},
            "kb": {"root_category": "ERSAAL"},
            "qdrant": {"collection_name": "ersaal_kb"},
            "docs": {"repo": "lamah-co/ersaal-docs", "site_url": SITE, "sections": sections},
        }
    )


ALL_SECTIONS = [
    {"path": "content/docs/faq.mdx", "kind": "faq", "category": "الأسئلة الشائعة"},
    {"path": "content/docs", "kind": "guide"},
    {"path": "content/api", "kind": "guide", "category": "API"},
]


def test_faq_gives_one_answer_per_question_under_its_group() -> None:
    """faq.mdx: one answer per question, under its group, separators removed, each with its source link."""
    plan = build_plan(ersaal(ALL_SECTIONS[:1]), FIXTURES)

    faq = [a for a in plan.articles if a.key.startswith("faq:")]
    assert len(faq) == 5
    groups = {a.category_path for a in faq}
    assert groups == {("الأسئلة الشائعة", "أسئلة عامة"), ("الأسئلة الشائعة", "أسئلة رسائل OTP")}
    assert sum(a.category_path[-1] == "أسئلة عامة" for a in faq) == 3
    first = faq[0]
    assert first.title == "ما هي المنصة؟"
    assert "<hr" not in first.html
    assert f'href="{SITE}/faq"' in first.html


def test_guide_page_resolves_steps_callouts_images_and_links() -> None:
    """A guide with Steps, Callouts, an image and a site-relative link becomes clean HTML on the docs site."""
    plan = build_plan(ersaal(ALL_SECTIONS), FIXTURES)

    page = next(a for a in plan.articles if a.key == "page:content/docs/senders/create")
    assert page.title == "طلب اسم مرسل جديد"
    assert page.category_path == ("أسماء المرسلين",)
    assert "<strong>1. الانتقال إلى أسماء المرسلين</strong>" in page.html
    assert "<blockquote>" in page.html and "تنبيه:" in page.html
    assert f'href="{SITE}/screenshots/senders-create.png">نموذج طلب اسم مرسل جديد</a>' in page.html
    assert f'href="{SITE}/senders/registration-requirement"' in page.html
    assert "<Step" not in page.html and "import " not in page.html
    assert page.source_url == f"{SITE}/senders/create"


def test_faq_file_is_not_synced_twice_by_the_guide_section() -> None:
    """The FAQ belongs to its own section, so the docs folder skips it."""
    plan = build_plan(ersaal(ALL_SECTIONS), FIXTURES)

    assert not [a for a in plan.articles if a.key == "page:content/docs/faq"]


def test_api_page_keeps_tables_and_code_under_the_api_category() -> None:
    """English API pages keep their tables and code, under API › SMS."""
    plan = build_plan(ersaal(ALL_SECTIONS), FIXTURES)

    page = next(a for a in plan.articles if a.key == "page:content/api/sms/send-message")
    assert page.category_path[0] == "API"
    assert "<table>" in page.html and "<code>" in page.html
    assert "Source:" in page.html


def test_code_blocks_keep_their_imports_and_tabs_become_headings() -> None:
    """Imports inside code examples stay; only MDX imports go. Tabs become bold labels."""
    mdx = (
        "---\ntitle: Quickstart\n---\n\nimport { Callout } from 'fumadocs-ui/components/callout';\n\n"
        "<Tabs items={['Python']}>\n<Tab value=\"Python\">\n```python\nimport requests\n```\n</Tab>\n</Tabs>\n"
    )

    page = parse_mdx(mdx)

    assert "import requests" in page.markdown
    assert "fumadocs-ui" not in page.markdown
    assert "**Python**" in page.markdown and "<Tab" not in page.markdown


def test_cards_become_a_link_list() -> None:
    """Cards turn into links with their description."""
    page = parse_mdx(
        '<Cards>\n  <Card title="المجموعات" href="/contacts/groups" description="إدارة المجموعات" />\n</Cards>\n'
    )

    html = markdown_to_html(page.markdown, SITE)

    assert f'<a href="{SITE}/contacts/groups">المجموعات</a> — إدارة المجموعات' in html


def test_unchanged_docs_give_identical_hashes() -> None:
    """Building twice from the same files gives the same hashes, so a re-sync writes nothing."""
    first = build_plan(ersaal(ALL_SECTIONS), FIXTURES)
    second = build_plan(ersaal(ALL_SECTIONS), FIXTURES)

    assert [a.content_hash for a in first.articles] == [a.content_hash for a in second.articles]


@pytest.mark.parametrize(
    ("relative", "url"),
    [
        ("content/docs/senders/create.mdx", f"{SITE}/senders/create"),
        ("content/docs/senders/index.mdx", f"{SITE}/senders"),
        ("content/api/sms/send-message.mdx", f"{SITE}/api/sms/send-message"),
    ],
)
def test_page_urls_follow_the_docs_site(relative: str, url: str) -> None:
    """Page URLs match the Fumadocs routes."""
    assert page_url(SITE, Path(relative)) == url


def test_load_projects_rejects_a_shared_collection(tmp_path: Path) -> None:
    """Two projects can never share a Qdrant collection."""
    for key, group in (("ersaal", "ERSAAL Support"), ("demo", "Demo Support")):
        (tmp_path / f"{key}.yaml").write_text(
            f"key: {key}\nzammad: {{group: {group}}}\nkb: {{root_category: {key}}}\nqdrant: {{collection_name: shared_kb}}\n",
            encoding="utf-8",
        )

    with pytest.raises(ValueError, match="Qdrant collection"):
        load_projects(tmp_path)
