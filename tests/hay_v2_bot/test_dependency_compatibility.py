"""Offline compatibility contract tests for the additive hay_v2_bot stack."""

from __future__ import annotations

import inspect

from haystack import Pipeline
from haystack.components.builders import ChatPromptBuilder
from haystack.components.embedders import OpenAIDocumentEmbedder, OpenAITextEmbedder
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
from haystack.components.writers import DocumentWriter
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.utils import Secret
from haystack_integrations.components.converters.docling import (
    DoclingConverter,
    ExportType,
)
from haystack_integrations.components.retrievers.pinecone import (
    PineconeEmbeddingRetriever,
)
from haystack_integrations.document_stores.pinecone import (
    PineconeDocumentStore,
)

API_BASE_URL = "https://example.invalid/v1"
CHAT_MODEL = "gpt-4o-mini"
EMBEDDING_MODEL = "text-embedding-3-small"


def _dummy_secret() -> Secret:
    return Secret.from_token("stage3-offline-secret")


def test_canonical_docling_and_pinecone_imports_succeed() -> None:
    assert inspect.isclass(DoclingConverter)
    assert ExportType.DOC_CHUNKS.name == "DOC_CHUNKS"
    assert inspect.isclass(PineconeDocumentStore)
    assert inspect.isclass(PineconeEmbeddingRetriever)


def test_docling_converter_constructs_for_doc_chunks_without_running() -> None:
    converter = DoclingConverter(export_type=ExportType.DOC_CHUNKS)

    assert isinstance(converter, DoclingConverter)


def test_openai_text_embedder_accepts_offline_configuration() -> None:
    embedder = OpenAITextEmbedder(
        api_key=_dummy_secret(),
        model=EMBEDDING_MODEL,
        api_base_url=API_BASE_URL,
    )

    assert isinstance(embedder, OpenAITextEmbedder)


def test_openai_document_embedder_accepts_offline_configuration() -> None:
    embedder = OpenAIDocumentEmbedder(
        api_key=_dummy_secret(),
        model=EMBEDDING_MODEL,
        api_base_url=API_BASE_URL,
    )

    assert isinstance(embedder, OpenAIDocumentEmbedder)


def test_openai_chat_generator_accepts_offline_configuration() -> None:
    generator = OpenAIChatGenerator(
        api_key=_dummy_secret(),
        model=CHAT_MODEL,
        api_base_url=API_BASE_URL,
    )

    assert isinstance(generator, OpenAIChatGenerator)


def test_offline_ingestion_pipeline_connects_converter_embedder_and_writer() -> None:
    pipeline = Pipeline()
    document_store = InMemoryDocumentStore()

    pipeline.add_component(
        "converter",
        DoclingConverter(export_type=ExportType.DOC_CHUNKS),
    )
    pipeline.add_component(
        "embedder",
        OpenAIDocumentEmbedder(
            api_key=_dummy_secret(),
            model=EMBEDDING_MODEL,
            api_base_url=API_BASE_URL,
        ),
    )
    pipeline.add_component("writer", DocumentWriter(document_store=document_store))

    pipeline.connect("converter.documents", "embedder.documents")
    pipeline.connect("embedder.documents", "writer.documents")

    assert pipeline.graph.has_edge("converter", "embedder")
    assert pipeline.graph.has_edge("embedder", "writer")


def test_offline_retrieval_pipeline_connects_embedder_and_retriever() -> None:
    pipeline = Pipeline()

    pipeline.add_component(
        "embedder",
        OpenAITextEmbedder(
            api_key=_dummy_secret(),
            model=EMBEDDING_MODEL,
            api_base_url=API_BASE_URL,
        ),
    )
    pipeline.add_component(
        "retriever",
        InMemoryEmbeddingRetriever(document_store=InMemoryDocumentStore()),
    )

    pipeline.connect("embedder.embedding", "retriever.query_embedding")

    assert pipeline.graph.has_edge("embedder", "retriever")


def test_offline_generation_pipeline_connects_prompt_builder_and_generator() -> None:
    pipeline = Pipeline()

    pipeline.add_component("prompt_builder", ChatPromptBuilder(template=[]))
    pipeline.add_component(
        "generator",
        OpenAIChatGenerator(
            api_key=_dummy_secret(),
            model=CHAT_MODEL,
            api_base_url=API_BASE_URL,
        ),
    )

    pipeline.connect("prompt_builder.prompt", "generator.messages")

    assert pipeline.graph.has_edge("prompt_builder", "generator")


def test_pinecone_document_store_constructor_exposes_required_parameters() -> None:
    parameters = inspect.signature(PineconeDocumentStore.__init__).parameters

    for name in ("index", "namespace", "dimension", "metric"):
        assert name in parameters


def test_pinecone_embedding_retriever_constructor_exposes_required_parameters() -> None:
    parameters = inspect.signature(PineconeEmbeddingRetriever.__init__).parameters

    for name in ("document_store", "top_k"):
        assert name in parameters
    assert {"filters", "filter_policy"} & set(parameters)
