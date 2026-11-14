"""Tests that the judge sees the retrieved documents' content, not only their titles."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.documents import Document

from app.answer.judge import JudgeResult, judge_message
from app.answer.knowledgebase import QdrantKBClient
from app.answer.retrieval import collect_retrieved_documents, judge_sources, record_retrieved_documents
from app.answer.service import AnswerService
from app.models.answer import AnswerCandidate, DocumentDict
from app.settings import ZammadAISettings
from app.settings.answer import JudgeSettings, JudgeThresholds, StringPromptConfig

ANSWER = (
    "تستغرق عملية اعتماد اسم المرسل عادةً من ثلاثة أيام إلى أسبوع، حسب سرعة استجابة مزودي الخدمة في المنطقة. "
    "يمكنك متابعة حالة الطلب من لوحة التحكم، وسيقوم الفريق المختص بإبلاغك فور اكتمال الإجراءات اللازمة لاعتماد الاسم."
)
SENDER_ID_DOC = Document(
    page_content="اعتماد اسم المرسل يستغرق من 3 أيام إلى أسبوع حسب استجابة مزودي الخدمة.",
    metadata={"title": "اسم المرسل", "url": "https://docs.example.test/sender-id"},
)


def test_collector_records_retrievals_only_inside_the_block_and_from_child_tasks() -> None:
    """Retrievals in tools run in child tasks; they must reach the collector, and nothing leaks outside it."""

    async def scenario() -> list[Document]:
        record_retrieved_documents([Document(page_content="outside")])
        with collect_retrieved_documents() as retrieved:
            await asyncio.create_task(asyncio.to_thread(lambda: None))
            await asyncio.create_task(_record_async(SENDER_ID_DOC))
        record_retrieved_documents([Document(page_content="after")])
        return retrieved

    assert [d.page_content for d in asyncio.run(scenario())] == [SENDER_ID_DOC.page_content]


async def _record_async(document: Document) -> None:
    record_retrieved_documents([document])


@pytest.mark.asyncio
async def test_knowledge_base_search_records_its_results() -> None:
    """Every Qdrant search inside a collection is recorded, for the agent tool and direct RAG alike."""
    client = QdrantKBClient.__new__(QdrantKBClient)
    client.qdrant_settings = MagicMock(retrieval_num_documents=5)
    client.vectorstore = MagicMock()
    client.vectorstore.asimilarity_search_with_relevance_scores = AsyncMock(return_value=[(SENDER_ID_DOC, 0.91)])

    with collect_retrieved_documents() as retrieved:
        await client.asearch_documents(query="sender id")

    assert retrieved == [SENDER_ID_DOC]


def test_judge_sources_dedupes_truncates_and_limits() -> None:
    """Repeated retrievals appear once, long documents are cut, and at most max_documents are shown."""
    long_doc = Document(page_content="x" * 50, metadata={"title": "Long", "url": "https://docs.example.test/long"})
    other = Document(page_content="other", metadata={"title": "Other"})

    rendered = judge_sources([SENDER_ID_DOC, SENDER_ID_DOC, long_doc, other], max_documents=2, max_chars=10)

    assert rendered.count("اسم المرسل\nURL") == 1
    assert "x" * 10 + " […]" in rendered and "x" * 11 not in rendered
    assert "Other" not in rendered
    assert judge_sources([], max_documents=5, max_chars=10) == "No documents were retrieved."


def test_judge_message_contains_document_content_and_cited_titles() -> None:
    """The judge gets the text to check groundedness against, plus what the answer cites."""
    sources = judge_sources([SENDER_ID_DOC], max_documents=5, max_chars=4000)

    message = judge_message("كم يستغرق اعتماد اسم المرسل؟", ANSWER, [{"title": "اسم المرسل", "url": "u"}], sources)

    assert SENDER_ID_DOC.page_content in message
    assert "Cited by the answer: اسم المرسل" in message


class RecordingJudge:
    """Judge fake that records its inputs and passes."""

    def __init__(self) -> None:
        """Start with no calls."""
        self.calls: list[dict] = []

    async def judge_answer(self, **kwargs) -> JudgeResult:
        """Record the call and pass the answer."""
        self.calls.append(kwargs)
        return JudgeResult(context_relevance=1.0, groundedness=1.0, answer_relevance=1.0, passed=True, reasoning="ok")


@pytest.mark.asyncio
async def test_answer_service_gives_the_judge_what_the_agent_retrieved() -> None:
    """Documents the agent's tool retrieved during generation reach the judge as content."""

    async def agent_run(*_args, **_kwargs) -> dict:
        record_retrieved_documents([SENDER_ID_DOC])  # what the knowledge-base tool does
        return {
            "structured_response": AnswerCandidate(
                response=ANSWER, documents=[DocumentDict(title="اسم المرسل", url="https://docs.example.test/sender-id")]
            )
        }

    service = AnswerService.__new__(AnswerService)
    service.settings = ZammadAISettings.model_construct()
    service.settings.answer = MagicMock(ai_answer_disclaimer="")
    service.langfuse_client = None
    service.user_message_template = MagicMock(format=lambda **kwargs: "question")
    service.answer_strategy = "agent"
    service.agent = MagicMock(ainvoke=agent_run)
    service.agent_context = object()
    judge = RecordingJudge()
    service.judge_handler = judge
    service.judge_settings = JudgeSettings(
        enabled=True,
        thresholds=JudgeThresholds(),
        prompt=StringPromptConfig(prompt="Judge."),
        repair_prompt=StringPromptConfig(prompt="Repair: {repair_instructions}"),
    )

    await service.generate_answer(user_text="كم يستغرق اعتماد اسم المرسل؟", category="FAQ")

    assert SENDER_ID_DOC.page_content in judge.calls[0]["sources"]
