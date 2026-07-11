"""Offline tests for hay_v2_bot.pipelines.factory."""

from __future__ import annotations

from hay_v2_bot.config import DocumentRagSettings
from hay_v2_bot.models import INSUFFICIENT_DOCUMENT_ANSWER
from hay_v2_bot.pipelines import (
    build_ingestion_pipeline,
    build_rag_pipeline,
    build_summary_pipeline,
)
from haystack.document_stores.types import DuplicatePolicy
from haystack.utils import Secret
from haystack_integrations.document_stores.pinecone import PineconeDocumentStore


def _settings() -> DocumentRagSettings:
    return DocumentRagSettings(
        _env_file=None,
        PINECONE_API_KEY="pinecone-key",
        PINECONE_INDEX_NAME="document-index",
        OPENAI_API_KEY="openai-key",
        OPENAI_BASE_URL="https://example.invalid/v1",
        OPENAI_EMBEDDING_MODEL="embedding-model",
        OPENAI_CHAT_MODEL="chat-model",
        embedding_dimensions=1536,
        retrieval_top_k=4,
    )


def _document_store() -> PineconeDocumentStore:
    return PineconeDocumentStore(
        api_key=Secret.from_token("pinecone-key"),
        index="document-index",
        namespace="telegram-documents-user-123",
        dimension=1536,
        metric="cosine",
    )


def test_ingestion_pipeline_builds_with_verified_wiring_and_overwrite_policy() -> None:
    settings = _settings()
    store = _document_store()

    pipeline = build_ingestion_pipeline(settings, store)
    inputs = pipeline.inputs()
    embedder = pipeline.get_component("embedder")
    writer = pipeline.get_component("writer")

    assert inputs["embedder"]["documents"]["is_mandatory"] is True
    assert "documents" not in inputs["writer"]
    assert embedder.model == "embedding-model"
    assert embedder.api_base_url == "https://example.invalid/v1"
    assert embedder.dimensions == 1536
    assert embedder.progress_bar is False
    assert writer.policy is DuplicatePolicy.OVERWRITE
    assert writer.document_store is store


def test_summary_pipeline_builds_with_russian_single_sentence_prompt_and_deterministic_generator(
) -> None:
    settings = _settings()

    pipeline = build_summary_pipeline(settings)
    inputs = pipeline.inputs()
    prompt_builder = pipeline.get_component("prompt_builder")
    generator = pipeline.get_component("generator")
    system_text = prompt_builder.template[0].texts[0]
    user_text = prompt_builder.template[1].texts[0]

    assert inputs["prompt_builder"]["file_name"]["is_mandatory"] is True
    assert inputs["prompt_builder"]["document_context"]["is_mandatory"] is True
    assert prompt_builder.required_variables == ["file_name", "document_context"]
    assert "exactly one concise sentence in Russian" in system_text
    assert "Preserve names, dates, numbers, currencies, and technical terms" in system_text
    assert "source document is in English" in system_text
    assert "natural Russian" in system_text
    assert INSUFFICIENT_DOCUMENT_ANSWER not in system_text
    assert "{{ document_context }}" in user_text
    assert generator.model == "chat-model"
    assert generator.api_base_url == "https://example.invalid/v1"
    assert generator.generation_kwargs == {"temperature": 0}


def test_rag_pipeline_builds_with_russian_answer_contract_and_exact_fallback_prompt() -> None:
    settings = _settings()
    store = _document_store()

    pipeline = build_rag_pipeline(settings, store)
    inputs = pipeline.inputs()
    text_embedder = pipeline.get_component("text_embedder")
    retriever = pipeline.get_component("retriever")
    prompt_builder = pipeline.get_component("prompt_builder")
    generator = pipeline.get_component("generator")
    system_text = prompt_builder.template[0].texts[0]

    assert inputs["text_embedder"]["text"]["is_mandatory"] is True
    assert inputs["prompt_builder"]["question"]["is_mandatory"] is True
    assert prompt_builder.required_variables == ["question", "documents"]
    assert text_embedder.model == "embedding-model"
    assert text_embedder.api_base_url == "https://example.invalid/v1"
    assert text_embedder.dimensions == 1536
    assert retriever.document_store is store
    assert retriever.top_k == 4
    assert retriever.filters == {
        "field": "record_type",
        "operator": "==",
        "value": "document_chunk",
    }
    assert "Answer only from the retrieved documents." in system_text
    assert "Answer in Russian even if the retrieved documents are in English." in system_text
    assert "Preserve exact names, dates, numbers, currencies, and units." in system_text
    assert INSUFFICIENT_DOCUMENT_ANSWER in system_text
    assert generator.model == "chat-model"
    assert generator.api_base_url == "https://example.invalid/v1"
    assert generator.generation_kwargs == {"temperature": 0}


def test_factories_use_fresh_component_instances() -> None:
    settings = _settings()
    store = _document_store()

    first_ingestion = build_ingestion_pipeline(settings, store)
    second_ingestion = build_ingestion_pipeline(settings, store)
    first_summary = build_summary_pipeline(settings)
    second_summary = build_summary_pipeline(settings)
    first_rag = build_rag_pipeline(settings, store)
    second_rag = build_rag_pipeline(settings, store)

    assert first_ingestion.get_component("embedder") is not second_ingestion.get_component(
        "embedder"
    )
    assert first_ingestion.get_component("writer") is not second_ingestion.get_component("writer")
    assert first_summary.get_component("prompt_builder") is not second_summary.get_component(
        "prompt_builder"
    )
    assert first_summary.get_component("generator") is not second_summary.get_component(
        "generator"
    )
    assert first_rag.get_component("text_embedder") is not second_rag.get_component(
        "text_embedder"
    )
    assert first_rag.get_component("retriever") is not second_rag.get_component("retriever")
    assert first_rag.get_component("prompt_builder") is not second_rag.get_component(
        "prompt_builder"
    )
    assert first_rag.get_component("generator") is not second_rag.get_component("generator")
