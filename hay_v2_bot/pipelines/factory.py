"""Fresh Haystack pipeline factories for document RAG workflows."""

from __future__ import annotations

from haystack import Pipeline
from haystack.components.builders import ChatPromptBuilder
from haystack.components.embedders import OpenAIDocumentEmbedder, OpenAITextEmbedder
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.components.writers import DocumentWriter
from haystack.dataclasses import ChatMessage
from haystack.document_stores.types import DuplicatePolicy
from haystack.utils import Secret
from haystack_integrations.components.retrievers.pinecone import PineconeEmbeddingRetriever
from haystack_integrations.document_stores.pinecone import PineconeDocumentStore

from hay_v2_bot.components.answer_contract import DOCUMENT_LABEL_PREFIX
from hay_v2_bot.config import DocumentRagSettings

_DOCUMENT_FILTER = {"field": "record_type", "operator": "==", "value": "document_chunk"}


def _openai_secret(settings: DocumentRagSettings) -> Secret:
    return Secret.from_token(settings.OPENAI_API_KEY.get_secret_value())


def build_ingestion_pipeline(
    settings: DocumentRagSettings,
    document_store: PineconeDocumentStore,
) -> Pipeline:
    """Build a fresh embedding-and-write pipeline for normalized documents."""
    pipeline = Pipeline()
    pipeline.add_component(
        "embedder",
        OpenAIDocumentEmbedder(
            api_key=_openai_secret(settings),
            model=settings.OPENAI_EMBEDDING_MODEL,
            dimensions=settings.embedding_dimensions,
            api_base_url=settings.OPENAI_BASE_URL,
            progress_bar=False,
        ),
    )
    pipeline.add_component(
        "writer",
        DocumentWriter(document_store=document_store, policy=DuplicatePolicy.OVERWRITE),
    )
    pipeline.connect("embedder.documents", "writer.documents")
    return pipeline


def build_summary_pipeline(settings: DocumentRagSettings) -> Pipeline:
    """Build a fresh one-sentence summary pipeline."""
    pipeline = Pipeline()
    pipeline.add_component(
        "prompt_builder",
        ChatPromptBuilder(
            template=[
                ChatMessage.from_system(
                    "You summarize only the supplied document. "
                    "Return exactly one concise sentence in Russian. "
                    "Preserve names, dates, numbers, currencies, and technical terms "
                    "accurately. "
                    "Do not add unsupported facts. "
                    "If the source document is in English, translate its meaning into "
                    "natural Russian instead of copying an English sentence."
                ),
                ChatMessage.from_user(
                    "File name: {{ file_name }}\n\n"
                    "Document chunks:\n{{ document_context }}"
                ),
            ],
            required_variables=["file_name", "document_context"],
        ),
    )
    pipeline.add_component(
        "generator",
        OpenAIChatGenerator(
            api_key=_openai_secret(settings),
            model=settings.OPENAI_CHAT_MODEL,
            api_base_url=settings.OPENAI_BASE_URL,
            generation_kwargs={"temperature": 0},
        ),
    )
    pipeline.connect("prompt_builder.prompt", "generator.messages")
    return pipeline


def build_rag_pipeline(
    settings: DocumentRagSettings,
    document_store: PineconeDocumentStore,
) -> Pipeline:
    """Build a fresh grounded retrieval-and-answer pipeline."""
    pipeline = Pipeline()
    pipeline.add_component(
        "text_embedder",
        OpenAITextEmbedder(
            api_key=_openai_secret(settings),
            model=settings.OPENAI_EMBEDDING_MODEL,
            dimensions=settings.embedding_dimensions,
            api_base_url=settings.OPENAI_BASE_URL,
        ),
    )
    pipeline.add_component(
        "retriever",
        PineconeEmbeddingRetriever(
            document_store=document_store,
            top_k=settings.retrieval_top_k,
            filters=_DOCUMENT_FILTER,
        ),
    )
    pipeline.add_component(
        "prompt_builder",
        ChatPromptBuilder(
            template=[
                ChatMessage.from_system(
                    "Answer only from the retrieved documents. "
                    "Answer in Russian even if the retrieved documents are in English. "
                    "Preserve exact names, dates, numbers, currencies, and units. "
                    "Do not use unsupported external facts. "
                    "Each retrieved chunk is labelled with an id such as "
                    f"{DOCUMENT_LABEL_PREFIX}1. "
                    "Reply with exactly one JSON object and nothing else: no Markdown, "
                    "no code fences, no commentary. "
                    "It must have exactly these keys: "
                    '"answerable" (true only if the retrieved chunks contain enough '
                    "information to answer the question, otherwise false), "
                    '"answer" (the answer as a string when answerable is true, otherwise '
                    'null), and "source_ids" (the ids of the chunks the answer is based '
                    "on when answerable is true, otherwise an empty list). "
                    "List only ids that appear in the retrieved chunks, and do not "
                    "mention chunk ids inside the answer text. "
                    "If the evidence is insufficient, set answerable to false."
                ),
                ChatMessage.from_user(
                    "Question: {{ question }}\n\n"
                    "Retrieved chunks:\n"
                    "{% for document in documents %}"
                    "[" + DOCUMENT_LABEL_PREFIX + "{{ loop.index }} "
                    "| file={{ document.meta.file_name }} "
                    "| chunk={{ document.meta.chunk_index }}"
                    "{% if document.meta.page_number is defined "
                    "and document.meta.page_number is not none %}"
                    " | page={{ document.meta.page_number }}"
                    "{% endif %}]\n"
                    "{{ document.content }}\n\n"
                    "{% endfor %}"
                ),
            ],
            required_variables=["question", "documents"],
        ),
    )
    pipeline.add_component(
        "generator",
        OpenAIChatGenerator(
            api_key=_openai_secret(settings),
            model=settings.OPENAI_CHAT_MODEL,
            api_base_url=settings.OPENAI_BASE_URL,
            generation_kwargs={"temperature": 0},
        ),
    )
    pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
    pipeline.connect("retriever.documents", "prompt_builder.documents")
    pipeline.connect("prompt_builder.prompt", "generator.messages")
    return pipeline
