# Knowledge Base Indexer (zammad-ai-index)

A batch job that keeps Qdrant in line with the Zammad knowledge base, and syncs each project's docs repository into that knowledge base. It runs once and exits; `ZAMMAD_AI_COMMAND` (or `--command`) selects what it does.

| Command | What it does |
|---|---|
| `index` (default) | Embeds new and changed KB answers into Qdrant and removes deleted ones. |
| `sync-docs` | Syncs one project's docs repo into its KB category tree. |
| `archive-category` | Archives every answer in one category, for retiring hand-written answers the docs now provide. |

## Projects

With `ZAMMAD_AI_PROJECTS_DIR` set, every `<key>.yaml` in that folder is a project. A project has its own Zammad group, KB root category, Qdrant collection and, optionally, a docs repo:

```yaml
key: ersaal
zammad: { group: ERSAAL Support, adopt_groups: [ERSAAL] }
kb: { root_category: ERSAAL }
qdrant: { collection_name: ersaal_kb }
docs:
  repo: lamah-co/ersaal-docs
  branch: main
  site_url: https://docs.getersaal.com
  sections:
    - { path: content/docs/faq.mdx, kind: faq, category: الأسئلة الشائعة }
    - { path: content/docs, kind: guide }
    - { path: content/api, kind: guide, category: API }
```

**Indexing:** `index` indexes each project's category tree into its own collection, or only `--project <key>`. Every vector carries `metadata.project`, with a keyword payload index on it. Projects never share a collection, so the deletion step of one project can't remove another's points. Without `ZAMMAD_AI_PROJECTS_DIR`, `index` behaves as before: one collection, with optional `zammad.category_ids`.

## Docs Sync

`sync-docs --project <key>`:

1. Downloads the branch as a tarball through the GitHub API (`ZAMMAD_AI_DOCS_GITHUB_TOKEN`, read-only), and logs the commit SHA.
2. Converts the Fumadocs MDX pages:
   - `faq` sections: one answer per `###` question, under its `##` group.
   - `guide` sections: one answer per page, with folders as subcategories, in `meta.json` order.
3. Creates, updates, publishes and archives answers below the project's root category only, using `ZAMMAD_AI_KB_EDITOR_TOKEN` (a user with `knowledge_base.editor` only). Answers outside that tree are never changed.

**Conversion:**

- `Steps` become a numbered list, `Callout` a labelled quote, `Cards` a link list, and `Tabs` bold headings.
- Code blocks are left as they are.
- Site-relative links point at `site_url`. Screenshots become links, because Zammad removes external images.
- Each answer ends with its source link.

**How answers are matched:** by category path and title, read from Zammad on every run.

- **Change detection:** content hashes in `ZAMMAD_AI_SYNC_STATE_DIR/<key>.json` let unchanged answers be skipped, so a run without docs changes writes nothing. If the file is lost, the next run updates everything once.
- **Renames:** renaming a page title archives the old answer and creates a new one.

**Options:**

- `--dry-run true` prints the plan and writes nothing.
- `--then-index true` indexes the project afterwards, if the KB changed.

## Archiving Hand-Written Answers

`archive-category --archive-category-id <id>` archives every answer in that category and below it. Run it with `--dry-run true` first. Archived answers stay restorable in Zammad.
