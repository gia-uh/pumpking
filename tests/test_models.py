import uuid
import pytest
from pumpking.models import ChunkNode, DocumentRoot, ChunkPayload

def test_chunk_node_defaults():
    node = ChunkNode(content="Hello World")
    assert isinstance(node.id, uuid.UUID)
    assert node.content == "Hello World"
    assert node.parent_id is None
    assert node.content_raw is None
    assert node.annotations == {}

def test_chunk_node_optional_content():
    node = ChunkNode(content=None, annotations={"entity": "Elon"})
    assert node.content is None
    output = node.to_dict()
    assert "content" not in output
    assert output["annotations"] == {"entity": "Elon"}

def test_content_raw_optimization():
    node = ChunkNode(content="Text", content_raw="Text")
    assert node.content_raw is None

    node_diff = ChunkNode(content="Clean text", content_raw="<p>Clean text</p>")
    assert node_diff.content_raw == "<p>Clean text</p>"

def test_sparse_serialization_empty_annotations():
    node = ChunkNode(content="Sparse Check")
    output = node.to_dict()
    
    assert "annotations" not in output
    assert "content_raw" not in output
    assert "parent_id" not in output
    assert output["content"] == "Sparse Check"

def test_sparse_serialization_with_data():
    node = ChunkNode(
        content="Data", 
        annotations={"sentiment": 0.9}
    )
    output = node.to_dict()
    
    assert "annotations" in output
    assert output["annotations"] == {"sentiment": 0.9}

def test_document_root_structure():
    child1 = ChunkNode(content="Child 1")
    child2 = ChunkNode(content="Child 2", annotations={"idx": 2})
    
    doc = DocumentRoot(
        document="Full original text content",
        original_filename="test.txt", 
        children=[child1, child2]
    )
    output = doc.to_dict()

    assert output["original_filename"] == "test.txt"
    assert output["document"] == "Full original text content"
    assert len(output["children"]) == 2
    assert "annotations" not in output["children"][0]
    assert output["children"][1]["annotations"] == {"idx": 2}
    assert "metadata" not in output