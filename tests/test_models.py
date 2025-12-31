import uuid
from pumpking.models import (
    ChunkNode,
    ChunkPayload,
    DocumentRoot,
    EntityChunkPayload,
    EntityChunkNode,
    TopicChunkPayload,
    TopicChunkNode,
    ContextualChunkPayload,
    ContextualChunkNode,
)


def test_chunk_payload_logic():
    """
    Tests that content_raw is removed only when it is identical to content,
    and remains present when it contains structural noise like Markdown.
    """
    payload_identical = ChunkPayload(content="Clean text", content_raw="Clean text")
    data_identical = payload_identical.to_dict()
    assert "content_raw" not in data_identical

    payload_with_noise = ChunkPayload(
        content="Important heading",
        content_raw="# Important heading",
        annotations={"level": 1},
    )
    data_noise = payload_with_noise.to_dict()
    assert data_noise["content"] == "Important heading"
    assert data_noise["content_raw"] == "# Important heading"


def test_chunk_payload_semantic_nesting():
    """
    Tests that ChunkPayload maintains internal semantic hierarchy
    through the children field with realistic raw data.
    """
    child_payload = ChunkPayload(content="First item", content_raw="* First item")
    parent_payload = ChunkPayload(
        content="List of items",
        content_raw="### List of items",
        children=[child_payload],
    )
    data = parent_payload.to_dict()

    assert len(data["children"]) == 1
    assert data["children"][0]["content_raw"] == "* First item"


def test_chunk_node_separation_of_concerns():
    """
    Verifies that ChunkNode separates execution results from
    further processing branches without redundant text fields.
    """
    result_payload = ChunkPayload(
        content="Paragraph text", content_raw="> Paragraph text"
    )
    branch_node = ChunkNode(strategy_label="sentence_splitter")

    node = ChunkNode(
        strategy_label="paragraph_processor",
        results=[result_payload],
        branches=[branch_node],
    )

    data = node.to_dict()

    assert "content" not in data
    assert "content_raw" not in data
    assert data["results"][0]["content_raw"] == "> Paragraph text"
    assert data["branches"][0]["strategy_label"] == "sentence_splitter"


def test_specialized_nodes_and_payloads():
    """
    Tests the integrity of specialized nodes ensuring they maintain
    specific metadata and their result payloads with Markdown examples.
    """
    entity_payload = EntityChunkPayload(
        content="Apple Inc.",
        content_raw="**Apple Inc.**",
        entity="Apple Inc.",
        type="ORG",
    )
    entity_node = EntityChunkNode(
        entity="Apple Inc.", type="ORG", results=[entity_payload]
    )

    topic_payload = TopicChunkPayload(
        content="Technology Section",
        content_raw="== Technology Section ==",
        topic="Technology",
    )
    topic_node = TopicChunkNode(topic="Technology", results=[topic_payload])

    context_payload = ContextualChunkPayload(
        content="The quarterly report shows growth.",
        content_raw="_The quarterly report shows growth._",
        context="Annual General Meeting",
    )
    context_node = ContextualChunkNode(
        context="Annual General Meeting", results=[context_payload]
    )

    e_data = entity_node.to_dict()
    t_data = topic_node.to_dict()
    c_data = context_node.to_dict()

    assert e_data["entity"] == "Apple Inc."
    assert e_data["type"] == "ORG"
    assert e_data["results"][0]["content_raw"] == "**Apple Inc.**"

    assert t_data["topic"] == "Technology"
    assert t_data["results"][0]["content_raw"] == "== Technology Section =="

    assert c_data["context"] == "Annual General Meeting"
    assert c_data["results"][0]["content_raw"] == "_The quarterly report shows growth._"


def test_document_root_structure():
    """
    Ensures that DocumentRoot acts as the starting point for execution
    branches and handles original document formatting.
    """
    raw_doc = "# Document Title\n\nContent here."
    root = DocumentRoot(document=raw_doc, original_filename="test.md")
    first_step_node = ChunkNode(strategy_label="header_cleaner")
    root.branches.append(first_step_node)

    data = root.to_dict()

    assert data["document"] == raw_doc
    assert data["original_filename"] == "test.md"
    assert data["branches"][0]["strategy_label"] == "header_cleaner"


def test_recursive_id_assignment():
    """
    Validates unique UUID assignment and parent-id references across the graph.
    """
    root = DocumentRoot(document="source")
    node = ChunkNode(parent_id=root.id, strategy_label="step_1")

    assert isinstance(root.id, uuid.UUID)
    assert node.parent_id == root.id
    assert node.id != root.id
