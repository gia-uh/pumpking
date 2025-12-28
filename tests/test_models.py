import uuid
import pytest
from pumpking.models import ChunkNode, DocumentRoot

def test_chunk_node_defaults():
    """Test that ID is generated and defaults are set correctly."""
    node = ChunkNode(content="Hello World")
    assert isinstance(node.id, uuid.UUID)
    assert node.content == "Hello World"
    assert node.parent_id is None
    assert node.content_raw is None
    assert node.annotations == {}

def test_content_raw_optimization():
    """Test that content_raw is set to None if it matches content."""
    # Case 1: Identical content -> content_raw become None
    node = ChunkNode(content="Text", content_raw="Text")
    assert node.content_raw is None

    # Case 2: Different content -> content_raw is preserved
    node_diff = ChunkNode(content="Clean text", content_raw="<p>Clean text</p>")
    assert node_diff.content_raw == "<p>Clean text</p>"

def test_sparse_serialization_empty_annotations():
    """Test that empty annotations are removed from the serialized output."""
    node = ChunkNode(content="Sparse Check")
    # By default annotations is {}, so it should disappear
    output = node.to_dict()
    
    assert "annotations" not in output
    assert "content_raw" not in output
    assert "parent_id" not in output
    assert output["content"] == "Sparse Check"

def test_sparse_serialization_with_data():
    """Test that populated fields remain."""
    node = ChunkNode(
        content="Data", 
        annotations={"sentiment": 0.9}
    )
    output = node.to_dict()
    
    assert "annotations" in output
    assert output["annotations"] == {"sentiment": 0.9}

def test_document_root_structure():
    """Test DocumentRoot nesting and serialization."""
    child1 = ChunkNode(content="Child 1")
    child2 = ChunkNode(content="Child 2", annotations={"idx": 2})
    
    doc = DocumentRoot(original_filename="test.txt", children=[child1, child2])
    output = doc.to_dict()

    assert output["original_filename"] == "test.txt"
    assert len(output["children"]) == 2
    # Check that child1 has no 'annotations' key (sparse)
    assert "annotations" not in output["children"][0]
    # Check that child2 has 'annotations' key
    assert output["children"][1]["annotations"] == {"idx": 2}
    # Metadata should be gone because it's empty
    assert "metadata" not in output