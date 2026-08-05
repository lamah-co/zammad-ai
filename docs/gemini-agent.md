# Native Gemini Agent Configuration

Use the native Gemini provider for the standard answer agent. It uses
`langchain-google-genai` for chat and embeddings so Gemini tool-call metadata
remains attached to the original LangChain `AIMessage` during replay.

## Credentials

For the Gemini Developer API, inject this secret at runtime:

```env
GOOGLE_API_KEY=<secret>
```

For Vertex AI, use workload identity or approved Google Cloud credentials and:

```env
GOOGLE_GENAI_USE_VERTEXAI=true
GOOGLE_CLOUD_PROJECT=<project-id>
GOOGLE_CLOUD_LOCATION=us-central1
```

Do not commit these values. `OPENAI_BASE_URL` is not used by the native Gemini
adapter.

## Workflow Configuration

Gemini 2.5 example:

```yaml
genai:
  sdk: gemini
  max_retries: 3
  chat_model: gemini-2.5-flash
  triage_model: gemini-2.5-flash
  answer_model: gemini-2.5-flash
  judge_model: gemini-2.5-flash
  embedding_model: gemini-embedding-001
  vertexai: false
  include_thoughts: false
  triage_thinking_budget: -1
  answer_thinking_budget: -1
  judge_thinking_budget: -1
  triage_temperature: 0.0
  answer_temperature: 0.0
  judge_temperature: 0.0
```

For Gemini 3, remove every `*_thinking_budget` value and configure supported
role-specific levels instead:

```yaml
genai:
  sdk: gemini
  chat_model: <approved-gemini-3-model>
  triage_thinking_level: low
  answer_thinking_level: low
  judge_thinking_level: low
```

The settings validator rejects a level and budget configured for the same role.
Keep `include_thoughts: false` unless reasoning summaries are explicitly needed
and approved for logs/traces.

## Index Configuration

The index and workflow must use the same embedding model and vector dimension:

```yaml
genai:
  sdk: gemini
  max_retries: 3
  embedding_model: gemini-embedding-001

qdrant:
  collection_name: your-collection
  vector_dimension: 768
```

The native adapter requests `output_dimensionality` equal to the configured
Qdrant dimension. Both services verify the returned vector length before use.
Changing model or dimension requires a new/rebuilt collection; do not mix vector
shapes in an existing collection.

## Local Verification

Run normal checks without a provider key:

```bash
cd zammad-ai-workflow
uv sync --dev
uv run pytest
uv run ruff format --check .
uv run ruff check .
uv run ty check

cd ../zammad-ai-index
uv sync --dev
uv run pytest
uv run ruff format --check .
uv run ruff check .
uv run ty check
```

The live contract tests are opt-in and must use a dedicated non-production key:

```bash
export GOOGLE_API_KEY='<test-secret>'
export ZAMMAD_AI_RUN_GEMINI_CONTRACT_TESTS=1
export ZAMMAD_AI_GEMINI_CONTRACT_MODEL=gemini-2.5-flash
uv run pytest test/integration/test_gemini_agent_contract.py -v
```

They verify both a direct tool-result replay and the `create_agent` retrieval
plus final `ToolStrategy` structured-response shape. The tests preserve the
original provider `AIMessage`; they do not log or fixture thought signatures.

## Migration From The Compatible Endpoint

1. Add `GOOGLE_API_KEY` or Vertex AI credentials to runtime secret storage.
2. Change workflow and index `genai.sdk` from `openai` to `gemini`.
3. Remove Gemini's URL and key from `OPENAI_BASE_URL` and `OPENAI_API_KEY`;
   reserve those variables for real OpenAI deployments.
4. Configure approved Gemini chat and embedding model IDs.
5. Confirm Qdrant vector dimension and rebuild the disposable index.
6. Run live contract tests, then real Zammad draft-mode Arabic tests.
7. Keep category `auto_publish: false` until channel and customer-end evidence
   is approved.

## Provider Boundary

`sdk: gemini` always selects `ChatGoogleGenerativeAI` and
`GoogleGenerativeAIEmbeddings`. `sdk: openai` selects the OpenAI adapters and
rejects Gemini model IDs. The workflow does not reconstruct Gemini messages or
implement provider-specific thought-signature handling; the native integration
preserves the original `AIMessage` during the agent tool loop.
