"""Shared test setup: modules with a module-level logger load the settings on import, so give them a Zammad config."""

import os

os.environ.setdefault(
    "ZAMMAD_AI_ZAMMAD",
    '{"type":"api","base_url":"https://zammad.example.com","knowledge_base_id":2,"auth_token":"test-token"}',
)
