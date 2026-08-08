"""Provider factories for index-job embedding models."""

from langchain_core.embeddings import Embeddings

from job.settings.genai import GenAIProviderSettings


def get_embedding_model(genai_settings: GenAIProviderSettings, vector_dimension: int) -> Embeddings:
    """Construct a provider-native embedding model with an explicit output size."""
    match genai_settings.sdk:
        case "gemini":
            from langchain_google_genai import GoogleGenerativeAIEmbeddings

            return GoogleGenerativeAIEmbeddings(
                model=genai_settings.embedding_model,
                output_dimensionality=vector_dimension,
                vertexai=False,
            )
        case "openai":
            from langchain_openai import OpenAIEmbeddings

            return OpenAIEmbeddings(
                model=genai_settings.embedding_model,
                dimensions=vector_dimension,
                max_retries=genai_settings.max_retries,
            )
        case _:
            raise ValueError(f"Unsupported GenAI SDK for embeddings: {genai_settings.sdk}")
