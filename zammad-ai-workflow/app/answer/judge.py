"""Judge generated answers with a structured LLM response and optional Langfuse tracing."""

from logging import Logger
from typing import Any
from uuid import uuid4

from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langfuse import propagate_attributes

from app.errors import TriageJudgeError, classify_provider_error
from app.models.answer import JudgeResult
from app.observe import LangfuseClient
from app.settings.genai import GenAIProviderSettings
from app.utils.genai_provider import get_chat_model
from app.utils.langchain import extract_structured_response
from app.utils.logging import getLogger

logger: Logger = getLogger("zammad-ai.answer.judge")


def judge_message(question: str, answer: str, documents: list[dict[str, Any]], sources: str | None) -> str:
    """Build the judge input: the retrieved documents' content, plus the titles the answer cites."""
    if sources is None:
        return f"Question: {question}\n\nAnswer: {answer}\n\nDocuments: {documents}"
    cited = ", ".join(str(document.get("title", "")) for document in documents) or "none"
    return (
        f"Question: {question}\n\nAnswer: {answer}\n\nCited by the answer: {cited}\n\nRetrieved documents:\n\n{sources}"
    )


class JudgeHandler:
    """Judge generated answers with a structured LLM response."""

    def __init__(
        self, genai_settings: GenAIProviderSettings, prompt: str, langfuse_client: LangfuseClient | None = None
    ) -> None:
        """Initialize the judge chain for the configured GenAI backend."""
        self.langfuse_client: LangfuseClient | None = langfuse_client

        if not prompt.strip():
            raise ValueError("Judge prompt cannot be empty.")

        # Use provider factory to construct the chat model.
        self.chat_model = get_chat_model(genai_settings, "judge")

        self._judge_agent = create_agent(
            model=self.chat_model,
            tools=[],
            system_prompt=(
                f"{prompt}\n\n"
                "When you are ready to produce the final judgment, call exactly one structured response tool "
                "for the JudgeResult schema. Do not return the judgment as free text, markdown, or raw JSON."
            ),
            response_format=ToolStrategy(
                schema=JudgeResult,
                tool_message_content="Answer judgment has been generated.",
            ),
        )

    async def judge_answer(
        self,
        question: str,
        answer: str,
        documents: list[dict[str, Any]],
        session_id: str | None = None,
        sources: str | None = None,
    ) -> JudgeResult:
        """Judge an answer against the retrieved documents' content and return the structured result.

        Args:
            question: The customer's question.
            answer: The answer to judge.
            documents: Documents the answer cites (title and URL).
            session_id: Optional Langfuse session ID.
            sources: The retrieved documents' content, rendered for the judge.
        """
        session_id, config = self._build_runnable_config(session_id=session_id)

        try:
            with propagate_attributes(session_id=session_id):
                agent_result: dict[str, Any] = await self._judge_agent.ainvoke(
                    input={"messages": [HumanMessage(content=judge_message(question, answer, documents, sources))]},
                    config=config,
                )
            return extract_structured_response(agent_result, JudgeResult)
        except Exception as e:
            logger.error("Error during judge invocation", exc_info=True)
            provider_error = classify_provider_error(e)
            raise TriageJudgeError("Judge operation failed", retryable=provider_error.retryable) from e

    def _build_runnable_config(self, session_id: str | None) -> tuple[str, RunnableConfig]:
        """Build a runnable config, creating a session id when needed."""
        resolved_session_id: str | None = session_id.strip() if session_id is not None else None
        if not resolved_session_id:
            resolved_session_id = str(uuid4())

        if self.langfuse_client is None:
            return resolved_session_id, RunnableConfig()

        config: RunnableConfig = self.langfuse_client.build_config(session_id=resolved_session_id)
        return resolved_session_id, config
