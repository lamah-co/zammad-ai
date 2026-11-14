"""Qdrant knowledge-base client and retrieval helpers."""

import asyncio
from logging import Logger
from uuid import NAMESPACE_DNS, UUID, uuid5

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_qdrant import QdrantVectorStore
from pydantic import BaseModel, Field, NonNegativeInt, PositiveInt
from qdrant_client import AsyncQdrantClient, QdrantClient
from qdrant_client.http.exceptions import ApiException
from qdrant_client.http.models import CollectionInfo
from qdrant_client.models import FieldCondition, Filter, MatchValue

from app.errors import QdrantPermanentError, QdrantRetryableError, classify_provider_error
from app.settings import QdrantSettings
from app.settings.genai import GenAIProviderSettings
from app.utils.genai_provider import get_embedding_model
from app.utils.logging import getLogger
from app.utils.status import provider_status, set_provider_ready

from .retrieval import record_retrieved_documents

logger: Logger = getLogger("zammad-ai.answer.knowledgebase")

# Create a consistent namespace UUID for generating vector IDs based on content
ZAMMAD_AI_NAMESPACE: UUID = uuid5(
    namespace=NAMESPACE_DNS,
    name="zammad-ai.muenchen.de",
)


class SearchQdrantKBInput(BaseModel):
    """Validated input for knowledge-base search queries."""

    query: str = Field(
        description="The search query string; should be concise and focused on the information needed; maximum length is 200 characters (~ 20 words).",
        max_length=200,
    )
    num_documents: PositiveInt = Field(
        default=5,
        description="The number of relevant documents to retrieve; should be a positive integer; default is 5.",
    )
    offset: NonNegativeInt = Field(
        default=0,
        description="The number of top relevant documents to skip for pagination; should be a non-negative integer; default is 0. Good for retrieving the next set of results in subsequent calls with the same query.",
    )


class RetrieveDocumentsKBOutput(BaseModel):
    """Knowledge-base search results with relevance scores."""

    documents_with_relevance_score: list[tuple[Document, float]] = Field(
        description="A list of tuples containing retrieved documents and their corresponding relevance scores between 0 and 1; the list is ordered by relevance score in descending order.",
    )


class QdrantKBError(QdrantPermanentError):
    """Custom exception for Qdrant-related errors."""

    ...


class QdrantKBClient:
    """Wrapper around Qdrant client to handle vector storage and retrieval."""

    def __init__(self, qdrant_settings: QdrantSettings, genai_settings: GenAIProviderSettings) -> None:
        # Create logger for QdrantClient
        """Initialize the QdrantKBClient, configure Qdrant clients, embeddings, vector store, and retriever.

        Parameters:
            qdrant_settings (QdrantSettings): Configuration for Qdrant connection, collection, vector dimensions, vector name, timeout, and retrieval defaults.
            genai_settings (GenAIProviderSettings): Configuration for the embedding provider (SDK, embedding model, max retries).

        Raises:
            QdrantKBError: If the configured Qdrant collection does not exist or is empty, if the GenAI SDK is unsupported, or if the embedding vector dimension does not match the configured Qdrant vector dimension.
        """
        self.logger: Logger = getLogger("zammad-ai.qdrant")

        self.collection_name: str = qdrant_settings.collection_name

        self.qdrant_settings: QdrantSettings = qdrant_settings
        # Create sync + async Qdrant client with appropriate configuration
        self.client = QdrantClient(
            url=qdrant_settings.url.encoded_string(),
            port=None,  # Port is included in the URL, so we set it to None
            timeout=qdrant_settings.timeout,
            api_key=qdrant_settings.api_key.get_secret_value() if qdrant_settings.api_key else None,
        )
        self.aclient = AsyncQdrantClient(
            url=qdrant_settings.url.encoded_string(),
            port=None,  # Port is included in the URL, so we set it to None
            timeout=qdrant_settings.timeout,
            api_key=qdrant_settings.api_key.get_secret_value() if qdrant_settings.api_key else None,
        )

        # Check if collection exists and if there is data in it, else raise an Error
        try:
            if not self.client.collection_exists(collection_name=qdrant_settings.collection_name):
                self.logger.error(f"Qdrant collection '{qdrant_settings.collection_name}' does not exist.")
                raise QdrantKBError(f"Qdrant collection '{qdrant_settings.collection_name}' does not exist.")

            collection_info: CollectionInfo = self.client.get_collection(
                collection_name=qdrant_settings.collection_name
            )
            if collection_info.points_count == 0:
                self.logger.warning(f"Qdrant collection '{qdrant_settings.collection_name}' exists but is empty.")

        except ApiException as e:
            self.logger.error("Error checking Qdrant collection existence or retrieving collection info", exc_info=True)
            status = getattr(e, "status", None)
            if isinstance(status, int) and status >= 500:
                raise QdrantRetryableError(
                    "Failed to check Qdrant collection existence or retrieve collection info"
                ) from e
            raise QdrantKBError("Failed to check Qdrant collection existence or retrieve collection info") from e

        # Create LangChain embedding model
        self.embeddings: Embeddings

        try:
            self.embeddings = get_embedding_model(genai_settings, qdrant_settings.vector_dimension)
        except (ImportError, ValueError) as error:
            self.logger.error(f"Unsupported GenAI SDK '{genai_settings.sdk}' for embeddings")
            raise QdrantKBError(f"Unsupported GenAI SDK '{genai_settings.sdk}' for embeddings") from error

        # Test embedding to ensure configuration is correct. A provider outage must not abort startup.
        provider_ready: bool = self.verify_embeddings()

        # Create LangChain Qdrant vector store. Its collection check embeds a test text,
        # so it only runs when the provider answered.
        self.vectorstore = QdrantVectorStore(
            client=self.client,
            collection_name=qdrant_settings.collection_name,
            embedding=self.embeddings,
            vector_name=qdrant_settings.vector_name,
            validate_collection_config=provider_ready,
        )

        self.retriever: VectorStoreRetriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": qdrant_settings.retrieval_num_documents},
        )

    def verify_embeddings(self) -> bool:
        """Check that the embedding provider answers with the configured vector dimension.

        Returns:
            bool: True when the provider answered, False when it is unavailable.

        Raises:
            QdrantKBError: If the provider answers with a different vector dimension, which is a configuration error.
        """
        expected: int = self.qdrant_settings.vector_dimension
        try:
            test_result: list[float] = self.embeddings.embed_query("This is a test string")
        except Exception as e:
            reason: str = type(classify_provider_error(e)).__name__
            self.logger.error(
                f"Embedding provider is unavailable ({reason}); starting without it and retrying in the background.",
                exc_info=True,
            )
            set_provider_ready(False, reason)
            return False

        if len(test_result) != expected:
            self.logger.error(
                f"Embedding dimension mismatch: expected {expected}, got {len(test_result)}. Check your GenAI embedding model configuration."
            )
            raise QdrantKBError(
                f"Embedding dimension mismatch: expected {expected}, got {len(test_result)}. Check your GenAI embedding model configuration."
            )
        set_provider_ready(True)
        return True

    async def asearch_documents(
        self,
        query: str,
        k: int | None = None,
        offset: int = 0,
        search_filter: Filter | None = None,
    ) -> list[tuple[Document, float]]:
        """Search for relevant documents in the Qdrant collection based on a query string.

        Args:
            query (str): The query string to search for relevant documents.
            k (int, optional): The number of top relevant documents to return.
            offset (int, optional): The number of top relevant documents to skip for pagination. Defaults to 0.
            search_filter (Filter, optional): Optional Qdrant metadata filter to scope retrieval.

        Returns:
            list[tuple[Document, float]]: A list of tuples containing relevant documents and their corresponding relevance scores between 0 and 1.
        """
        if k is None:
            k = self.qdrant_settings.retrieval_num_documents
        results = await self.vectorstore.asimilarity_search_with_relevance_scores(
            query=query,
            k=k,
            offset=offset,
            filter=search_filter,
        )
        record_retrieved_documents(document for document, _ in results)
        return results

    async def asearch_law_documents(
        self,
        law_id: str,
        query: str,
        k: int | None = None,
        offset: int = 0,
    ) -> list[tuple[Document, float]]:
        """Search indexed law chunks for one configured law."""
        law_filter = Filter(
            must=[
                FieldCondition(key="metadata.source", match=MatchValue(value="law")),
                FieldCondition(key="metadata.law_id", match=MatchValue(value=law_id)),
            ]
        )
        return await self.asearch_documents(
            query=query,
            k=k,
            offset=offset,
            search_filter=law_filter,
        )

    def search_documents(
        self,
        query: str,
        k: int | None = None,
        offset: int = 0,
        search_filter: Filter | None = None,
    ) -> list[tuple[Document, float]]:
        """Search for relevant documents in the Qdrant collection based on a query string.

        Args:
            query (str): The query string to search for relevant documents.
            k (int, optional): The number of top relevant documents to return.
            offset (int, optional): The number of top relevant documents to skip for pagination. Defaults to 0.
            search_filter (Filter, optional): Optional Qdrant metadata filter to scope retrieval.

        Returns:
            list[tuple[Document, float]]: A list of tuples containing relevant documents and their corresponding relevance scores between 0 and 1.
        """
        if k is None:
            k = self.qdrant_settings.retrieval_num_documents
        return self.vectorstore.similarity_search_with_relevance_scores(
            query=query,
            k=k,
            offset=offset,
            filter=search_filter,
        )

    async def close(self) -> None:
        """Close the Qdrant client connections."""
        await self.aclient.close()
        self.client.close()


PROVIDER_RETRY_INITIAL_SECONDS = 5.0
PROVIDER_RETRY_MAX_SECONDS = 60.0


async def watch_provider(qdrant_kb_client: QdrantKBClient) -> None:
    """Re-check the embedding provider with backoff until it answers, then mark it ready."""
    delay: float = PROVIDER_RETRY_INITIAL_SECONDS
    while not provider_status()[0]:
        await asyncio.sleep(delay)
        try:
            if await asyncio.to_thread(qdrant_kb_client.verify_embeddings):
                logger.info("Embedding provider is available again.")
                return
        except Exception:
            logger.error("Embedding provider check failed.", exc_info=True)
        delay = min(delay * 2, PROVIDER_RETRY_MAX_SECONDS)
