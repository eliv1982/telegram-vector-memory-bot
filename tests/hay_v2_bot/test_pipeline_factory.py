"""Offline tests for hay_v2_bot.pipelines.factory."""

from __future__ import annotations

from hay_v2_bot.components import parse_document_answer
from hay_v2_bot.config import DocumentRagSettings
from hay_v2_bot.models import INSUFFICIENT_DOCUMENT_ANSWER
from hay_v2_bot.pipelines import (
    build_ingestion_pipeline,
    build_rag_pipeline,
    build_summary_pipeline,
)
from haystack import Document
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


def test_rag_pipeline_builds_with_russian_answer_and_machine_readable_contract_prompt() -> None:
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
    # Routing is machine-readable: the prompt asks for a JSON contract, and the old
    # "return exactly this Russian sentence" instruction is gone.
    assert INSUFFICIENT_DOCUMENT_ANSWER not in system_text
    for contract_text in ('"answerable"', '"answer"', '"source_ids"', "JSON object", "DOC_1"):
        assert contract_text in system_text
    assert generator.model == "chat-model"
    assert generator.api_base_url == "https://example.invalid/v1"
    # No provider-specific structured-output option: the deterministic parser is the contract.
    assert generator.generation_kwargs == {"temperature": 0}


def _documents_for_prompt() -> list[Document]:
    return [
        Document(
            id="doc-" + "a" * 64 + "-chunk-000000",
            content="Первый фрагмент про бюджет.",
            meta={"file_name": "plan.pdf", "chunk_index": 0, "page_number": 3},
        ),
        Document(
            id="doc-" + "a" * 64 + "-chunk-000001",
            content="Второй фрагмент про сроки.",
            meta={"file_name": "plan.pdf", "chunk_index": 1},
        ),
        Document(
            id="doc-" + "b" * 64 + "-chunk-000000",
            content="Третий фрагмент.",
            meta={"file_name": "other.docx", "chunk_index": 0},
        ),
    ]


def test_rag_prompt_labels_retrieved_chunks_with_per_request_ids_in_order() -> None:
    pipeline = build_rag_pipeline(_settings(), _document_store())
    prompt_builder = pipeline.get_component("prompt_builder")

    messages = prompt_builder.run(question="Какой бюджет?", documents=_documents_for_prompt())[
        "prompt"
    ]
    user_text = messages[1].text

    assert "Question: Какой бюджет?" in user_text
    positions = [user_text.index(f"[DOC_{n} ") for n in (1, 2, 3)]
    assert positions == sorted(positions)
    assert "[DOC_4 " not in user_text
    assert "[DOC_1 | file=plan.pdf | chunk=0 | page=3]\nПервый фрагмент про бюджет." in user_text
    assert "[DOC_2 | file=plan.pdf | chunk=1]\nВторой фрагмент про сроки." in user_text
    assert "[DOC_3 | file=other.docx | chunk=0]\nТретий фрагмент." in user_text


def test_rag_prompt_does_not_expose_storage_ids_or_namespaces() -> None:
    pipeline = build_rag_pipeline(_settings(), _document_store())
    prompt_builder = pipeline.get_component("prompt_builder")

    messages = prompt_builder.run(question="Q?", documents=_documents_for_prompt())["prompt"]
    full_prompt = "\n".join(message.text for message in messages)

    assert "doc-" + "a" * 64 not in full_prompt
    assert "-chunk-0000" not in full_prompt
    assert "telegram-documents-user" not in full_prompt
    assert "pinecone-key" not in full_prompt
    assert "openai-key" not in full_prompt


def test_rag_system_prompt_survives_template_rendering_with_the_json_contract_intact() -> None:
    pipeline = build_rag_pipeline(_settings(), _document_store())
    prompt_builder = pipeline.get_component("prompt_builder")

    rendered_system = prompt_builder.run(question="Q?", documents=_documents_for_prompt())[
        "prompt"
    ][0].text

    assert rendered_system == prompt_builder.template[0].texts[0]
    assert "exactly one JSON object" in rendered_system


def test_prompt_labels_match_the_ids_the_parser_accepts() -> None:
    # The template and the parser share one label scheme (DOC_<1-based position>).
    pipeline = build_rag_pipeline(_settings(), _document_store())
    prompt_builder = pipeline.get_component("prompt_builder")
    documents = _documents_for_prompt()
    user_text = prompt_builder.run(question="Q?", documents=documents)["prompt"][1].text

    for position in range(1, len(documents) + 1):
        label = f"DOC_{position}"
        assert f"[{label} " in user_text
        reply = '{"answerable": true, "answer": "x", "source_ids": ["' + label + '"]}'
        assert parse_document_answer(reply, {"DOC_1", "DOC_2", "DOC_3"}).source_labels == (label,)


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
