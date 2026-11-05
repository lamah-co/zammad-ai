# Fork Maintenance

`lamah-co/zammad-ai` is a maintained fork of [`it-at-m/zammad-ai`](https://github.com/it-at-m/zammad-ai). Zammad remains the delivery engine; deployment, production configuration and prompts live in `lamah-co/zammad-infra`.

## Why the Fork Exists

Upstream posts answers as a bare article, which Zammad stores as a `note` that never reaches the customer. The fork replies on the customer's own channel, adds a native Gemini provider, and routes messages through Gemini moderation.

## Patch Register

| Patch | Docs | Upstream status | Drop from the fork when |
| --- | --- | --- | --- |
| Channel-aware replies and handoff | [channels.md](docs/components/channels.md) | Issue not filed yet | Upstream resolves the reply channel and falls back to a draft |
| Native Gemini provider and embeddings | [gemini-agent.md](docs/gemini-agent.md) | Compare with upstream during the next sync | Upstream provides an equivalent provider |
| Gemini moderation and localization | [moderation.md](docs/components/moderation.md) | Not proposed | Upstream provides equivalent routing |
| Direct RAG answer strategy | [configuration.md](docs/configuration.md) | Not proposed | No deployment uses `answer.strategy: direct_rag` |

Deployment policy, such as support prompts and category rules, stays in `lamah-co/zammad-infra` and is never proposed upstream.

## Syncing Upstream

Review upstream monthly, and immediately for security releases. Never auto-merge.

1. Fetch `origin` and `upstream`.
2. Create `chore/sync-upstream-YYYY-MM-DD` from `main`.
3. Merge `upstream/main`. Do not rebase shared branches.
4. Resolve conflicts one patch at a time, using the register above.
5. Run `uv run pytest`, `uv run ruff check .`, `uv run ruff format .` and `uv run ty check` in both services.
6. Build both images from the merge commit and run the end-to-end cases.
7. Open a PR listing the upstream SHA, the conflicts and any behaviour changes.
8. Update `lamah-co/zammad-infra` only after the PR is merged.

## Remotes

```text
origin    git@github.com:lamah-co/zammad-ai.git
upstream  https://github.com/it-at-m/zammad-ai.git
```
