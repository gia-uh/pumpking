import re
from io import StringIO
from html.parser import HTMLParser
import markdown

class _MLStripper(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.reset()
        self.strict = False
        self.convert_charrefs = True
        self.text = StringIO()

    def handle_data(self, d: str) -> None:
        self.text.write(d)

    def get_data(self) -> str:
        return self.text.getvalue()


def clean_text(
    text: str, 
    strip_markdown: bool = False, 
    collapse_whitespace: bool = True,
    to_lowercase: bool = False
) -> str:
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
    return re.sub(r'\s+', ' ', text)


def _strip_markdown_formatting(text: str) -> str:
    html = markdown.markdown(text)
    
    s = _MLStripper()
    s.feed(html)
    return s.get_data()