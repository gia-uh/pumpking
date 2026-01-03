import re
from io import StringIO
from html.parser import HTMLParser
import markdown

class _MLStripper(HTMLParser):
    """
    A specialized HTML parser designed to extract raw text content from markup.

    This class overrides the standard HTMLParser behavior to ignore tags and
    accumulate only the text data found between them. It serves as a helper
    component for the markdown stripping process, where text is first converted
    to HTML and then cleaned of tags using this stripper.
    """
    def __init__(self) -> None:
        super().__init__()
        self.reset()
        self.strict = False
        self.convert_charrefs = True
        self.text = StringIO()

    def handle_data(self, d: str) -> None:
        """
        Callback triggered when arbitrary data is encountered during parsing.
        
        Args:
            d: The content string found between tags.
        """
        self.text.write(d)

    def get_data(self) -> str:
        """
        Retrieves the accumulated text content from the parsed stream.

        Returns:
            The joined string of all text nodes encountered, effectively
            stripping out all HTML structure.
        """
        return self.text.getvalue()


def clean_text(
    text: str, 
    strip_markdown: bool = False, 
    collapse_whitespace: bool = True,
    to_lowercase: bool = False
) -> str:
    """
    The primary utility function for normalizing and sanitizing text strings.

    This function executes a configurable pipeline of cleaning operations. It is
    robust against null or empty inputs and guarantees that the output is free
    of leading or trailing whitespace.

    Args:
        text: The source string to process.
        strip_markdown: If True, renders markdown syntax to HTML and then strips
            the tags to return plain text.
        collapse_whitespace: If True, replaces all sequences of whitespace
            characters (including newlines and tabs) with a single space.
        to_lowercase: If True, converts all characters to lowercase.

    Returns:
        The cleaned and normalized string. Returns an empty string if the input
        was None or empty.
    """
    if not text:
        return ""

    cleaned = text

    if strip_markdown:
        cleaned = _strip_markdown_formatting(cleaned)

    if collapse_whitespace:
        cleaned = _collapse_extra_whitespace(cleaned)
    
    if to_lowercase:
        cleaned = cleaned.lower()

    return cleaned.strip()


def _collapse_extra_whitespace(text: str) -> str:
    """
    Internal helper that normalizes spacing within a string.

    Uses regular expressions to locate any sequence of whitespace characters
    (spaces, tabs, newlines) and replace them with a single space.

    Args:
        text: The string to normalize.

    Returns:
        The string with condensed whitespace.
    """
    return re.sub(r'\s+', ' ', text)


def _strip_markdown_formatting(text: str) -> str:
    """
    Internal helper that removes markdown syntax to produce plain text.

    This function utilizes a conversion strategy:
    1. It converts the Markdown input into HTML using the 'markdown' library.
    2. It feeds the resulting HTML into the _MLStripper to extract pure text,
       effectively removing formatting artifacts like bolding, headers, and links.

    Args:
        text: The string containing markdown syntax.

    Returns:
        The plain text content with all markdown formatting removed.
    """
    html = markdown.markdown(text)
    
    s = _MLStripper()
    s.feed(html)
    return s.get_data()