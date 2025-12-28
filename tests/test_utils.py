from pumpking.utils import clean_text

def test_clean_text_defaults():
    raw = "  Hello    World  "
    assert clean_text(raw) == "Hello World"

def test_clean_text_markdown_stripping_robust():
    raw = "## Title\nThis is **bold** and [link text](http://url)."
    
    result = clean_text(raw, strip_markdown=True)
    
    assert "Title" in result
    assert "bold" in result
    assert "link text" in result
    
    assert "**" not in result
    assert "http" not in result
    assert "##" not in result

def test_clean_text_all_options():
    raw = "## HELLO **WORLD** "
    
    result = clean_text(
        raw, 
        strip_markdown=True, 
        collapse_whitespace=True, 
        to_lowercase=True
    )
    
    assert result == "hello world"

def test_clean_text_preserves_structure_if_requested():
    raw = "Line 1\n\nLine 2"
    
    result = clean_text(raw, collapse_whitespace=False)
    
    assert result == "Line 1\n\nLine 2"