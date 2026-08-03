"""Opt-in live contract tests for Gemini multi-step tool calling."""

import os

import pytest
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain.messages import HumanMessage, ToolMessage
from langchain.tools import tool
from pydantic import BaseModel

from app.settings.genai import GenAIGeminiSettings
from app.utils.genai_provider import get_chat_model

pytestmark = pytest.mark.gemini_contract


def _live_contract_enabled() -> bool:
    has_key = bool(os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"))
    return os.getenv("ZAMMAD_AI_RUN_GEMINI_CONTRACT_TESTS") == "1" and has_key


@tool
def lookup_faq(query: str) -> str:
    """Return a deterministic FAQ answer for a contract test query."""
    return f"FAQ result for {query}: order tracking is available from the account page."


class ContractAnswer(BaseModel):
    """Minimal structured answer used to exercise LangChain's final response tool."""

    answer: str


@pytest.mark.skipif(not _live_contract_enabled(), reason="live Gemini contract test is not enabled")
async def test_gemini_replays_tool_call_with_provider_state() -> None:
    """The original Gemini AIMessage must survive the tool-result replay unchanged."""
    model_name = os.getenv("ZAMMAD_AI_GEMINI_CONTRACT_MODEL", "gemini-2.5-flash")
    model = get_chat_model(
        GenAIGeminiSettings(
            chat_model=model_name,
            max_retries=0,
            answer_thinking_budget=-1,
        ),
        "answer",
    )
    forced_model = model.bind_tools([lookup_faq], tool_choice="lookup_faq")
    automatic_model = model.bind_tools([lookup_faq], tool_choice="auto")
    user_message = HumanMessage(content="Use the FAQ tool to explain where an order can be tracked.")

    tool_request = await forced_model.ainvoke([user_message])

    assert len(tool_request.tool_calls) == 1
    tool_call = tool_request.tool_calls[0]
    tool_result = lookup_faq.invoke(tool_call["args"])
    final_response = await automatic_model.ainvoke(
        [
            user_message,
            tool_request,
            ToolMessage(content=str(tool_result), tool_call_id=tool_call["id"]),
        ]
    )

    assert final_response.text.strip()


@pytest.mark.skipif(not _live_contract_enabled(), reason="live Gemini contract test is not enabled")
async def test_gemini_langchain_agent_completes_tool_and_structured_output_loop() -> None:
    """The upstream create_agent shape must survive retrieval and final schema calls."""
    model_name = os.getenv("ZAMMAD_AI_GEMINI_CONTRACT_MODEL", "gemini-2.5-flash")
    model = get_chat_model(
        GenAIGeminiSettings(
            chat_model=model_name,
            max_retries=0,
            answer_thinking_budget=-1,
        ),
        "answer",
    )
    agent = create_agent(
        model=model,
        tools=[lookup_faq],
        system_prompt="Always call lookup_faq once, then return a ContractAnswer grounded in its result.",
        response_format=ToolStrategy(ContractAnswer),
    )

    result = await agent.ainvoke({"messages": [HumanMessage(content="Where can I track an order?")]})

    assert isinstance(result.get("structured_response"), ContractAnswer)
    assert result["structured_response"].answer.strip()
