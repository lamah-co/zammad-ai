# Configuration Guide

The Zammad-AI service is configured using a combination of YAML files, environment variables, and `.env` files.

## Configuration Hierarchy

The application loads configuration in the following order of priority (highest first):

1. **CLI Arguments**
2. **Environment Variables** (prefixed with `ZAMMAD_AI_`, e.g., `ZAMMAD_AI_GENAI__CHAT_MODEL`)
3. **.env File**
4. **config.yaml**
5. **Defaults** defined in the source code (`zammad-ai-workflow/app/settings/`)

## Core Configuration Sections

### Use Case

Identifies the specific deployment or purpose of the service.

- `usecase.name`: Machine-readable name.
- `usecase.description`: Human-readable description.

### GenAI

Configures the interface to the Large Language Model.

- `sdk`: Supports `openai`, `gemini`, and `anthropic` in the workflow. The
  index job supports `openai` and `gemini` embeddings.
- `chat_model`: The model used for generating responses (e.g., `gpt-4o`).
- `embedding_model`: The model used for vector search.
- `temperature`: Creativity setting (0.0 recommended for consistent results).

### Zammad

Connectivity settings for the Zammad instance.

- `type`: `api` (standard REST) or `eai` (Enterprise Application Integration).
- `base_url`: URL of your Zammad instance.
- `timeout`: Request timeout in seconds.
- `max_retries`: Number of retries for failed requests.

### Answer → Qdrant

Vector database settings for knowledge retrieval used by the answer generation pipeline.

- `url`: URL of the Qdrant instance.
- `api_key`: API key for authentication (set via env var).
- `collection_name`: The vector collection to query.
- `vector_name`: Optional name of the vector field.
- `vector_dimension`: Must match the embedding model output (e.g., 1024 or 1536).
- `timeout`: Timeout in seconds for Qdrant client operations.
- `retrieval_num_documents`: Number of documents to retrieve for context.

### Kafka

Event streaming configuration.

- `broker_url`: Kafka bootstrap server.
- `topic`: Topic to listen for new ticket events.
- `group_id`: Consumer group identifier.
- `security`: mTLS security settings (optional).

### Limits

- `max_user_text_length`: Maximum combined length of article text and attachment content accepted during triage and answer processing. Messages exceeding this length will be truncated to this limit.

### Triage

Defines the business logic for categorization and automation.

- `categories`: List of `id` and `name` pairs for classification.
- `actions`: List of available automated actions.
- `action_rules`: Mapping of categories and conditions to actions.
- `prompts`: Source of LLM prompt templates (`langfuse`, `file`, or `string`).
- `no_answer_public_fallback`, `human_review_public_fallback`: Customer acknowledgements for handoffs, as a string or a language map.
- `no_answer_internal_note`, `human_review_internal_note`: Internal note templates with `{category}`, `{action}`, `{customer_message}`, and `{reason}`. `no_action_internal_note` remains the fallback for no-answer handoffs.
- `no_answer_tag`, `human_review_tag`: Tags added to handoff tickets.
- `handoff_ticket_state`: Ticket state set on handoff. Leave unset to keep the state.

See [Channel-Aware Replies](components/channels.md) for publish gates and handoff behavior.

### Answer

- `strategy`: `agent` for tool calling, or `direct_rag` for one completion after Qdrant retrieval. `direct_rag` does not support `answer.dlf`.
- `direct_rag_prompt`: System prompt for the `direct_rag` strategy.

### Moderation

- `enabled`: Run Gemini moderation before triage and on generated answers.
- `provider`, `model`: Moderation provider and model. `model` falls back to `genai.judge_model`, then `genai.chat_model`.
- `block_on_unsafe`: Turn unsafe generated answers into shared drafts.
- `uncertain_action`: Route for uncertain results.
- `small_talk_category_name`: Category used for small talk.
- `out_of_scope_category_name`, `out_of_scope_action_name`: Category and `StaticAnswer` action used for out-of-scope messages.

See [Moderation and Localization](components/moderation.md).

### Localization

- `default_language`: Language used when none can be detected.
- `supported_languages`: Languages handled automatically (`ar`, `en`).
- `unsupported_language_action`: `human_review` or `default_language`.

### API

- `api_key`: Bearer key required by `/api/v1/triage` and `/api/v1/answer`. Requests are not authenticated when it is unset, so set `ZAMMAD_AI_API__API_KEY` in production. `/api/v1/health` stays open for container probes.

## Secrets Management

Sensitive information must NOT be placed in `config.yaml`. Use context-specific environment variables or a `.env` file:

```env
# OpenAI
OPENAI_API_KEY=sk-...

# Gemini Developer API (when genai.sdk is gemini)
GOOGLE_API_KEY=...

# Zammad
ZAMMAD_AI_ZAMMAD__AUTH_TOKEN=...
ZAMMAD_AI_ZAMMAD__RSS_FEED_TOKEN=...

# Qdrant
ZAMMAD_AI_QDRANT__API_KEY=...

# Langfuse (if prompts.type is "langfuse")
LANGFUSE_PUBLIC_KEY=...
LANGFUSE_SECRET_KEY=...
LANGFUSE_HOST=...
```

## Native Gemini Agent

Use `genai.sdk: gemini` when running Gemini models. The native adapter preserves
Gemini message metadata across LangChain tool calls. `sdk: openai` is reserved
for OpenAI models and rejects Gemini model IDs.

See [Gemini agent configuration](gemini-agent.md) for complete workflow/index
examples, migration, and contract tests.
