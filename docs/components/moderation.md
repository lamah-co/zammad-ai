# Moderation and Localization

Gemini moderation classifies each customer message before triage and each generated answer before publication. Enable it with `moderation.enabled`. It handles Arabic, English, Arabizi, dialect Arabic and mixed messages, and runs independently of the SLM guardrails.

## Customer Message Routing

| Decision | Route | Result |
| --- | --- | --- |
| `safe_support` | `continue_triage` | Normal triage |
| `small_talk` | `conversational_ai` | Category `moderation.small_talk_category_name`, using the action from its action rule |
| `out_of_scope` | `static_fallback` | Synthetic category `moderation.out_of_scope_category_name` with auto-publish, and the static action `moderation.out_of_scope_action_name` |
| `unsafe` | `human_review` | Handoff, see [Channel-Aware Replies](channels.md#handoff) |
| `uncertain` | `human_review` | Handoff |

Prompt injection and credential extraction attempts are classified as `unsafe`.

## Generated Answer Check

Each generated `AnswerCandidate` is moderated together with the customer message. When `moderation.block_on_unsafe` is true and the result routes to human review, `auto_publish` is cleared and the answer becomes a shared draft.

## Languages

Moderation also detects the customer language. `localization.supported_languages` lists the automated languages (`ar` and `en`).

- `unsupported_language_action: human_review` hands off a clearly unsupported language.
- `unsupported_language_action: default_language` continues in `localization.default_language`.

The detected language selects the localized text for static answers, public fallbacks and `triage.public_reply_subject`. Each must be a language map covering every supported language; otherwise the service refuses to start:

```yaml
triage:
  no_answer_public_fallback:
    ar: "تم استلام سؤالك، سنقوم بالرد عليك في أقرب وقت ممكن."
    en: "We received your question and will reply as soon as possible."
```

Generated answers receive the detected language as an explicit instruction. For an English customer, the model translates facts from the knowledge base into English before giving the final answer, and keeps URLs, numbers and product names unchanged. The judge treats language as out of scope, so a correct translation counts as grounded.
