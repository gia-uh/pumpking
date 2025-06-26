![pumpking](https://github.com/user-attachments/assets/2fd50356-7329-4415-9331-2287e061dad9)

# pumpking

> **P**rocessing, **U**nderstanding, and **M**odeling **P**ipeline for **K**nowledge **I**ntegration and **N**atural-language **G**eneration.

[](https://www.google.com/search?q=https://pypi.org/project/pumpking/)
[](https://www.google.com/search?q=https://github.com/YOUR_USERNAME/pumpking/actions)
[](https://opensource.org/licenses/MIT)
[](https://www.google.com/search?q=https://pypi.org/project/pumpking)

`pumpking` is an open-source Python library designed to streamline the complex process of document chunking, parsing, and representation. It provides a flexible and powerful pipeline to transform unstructured documents into structured, queryable knowledge, ready for use with Large Language Models (LLMs).

Whether you're building a RAG (Retrieval-Augmented Generation) system, a document analysis tool, or a knowledge extraction service, `pumpking` provides the foundational blocks to get you there faster.

## Key Features

  * **📄 Versatile Document Parsing**: Ingest and parse various document formats, including PDF, TXT, HTML, and more.
  * **🧩 Intelligent Chunking**: Go beyond fixed-size chunks. Use semantic, recursive, or agentic chunking strategies to preserve context.
  * **🧠 LLM-Powered Representation**: Automatically generate rich representations from your documents, such as vector embeddings or knowledge graphs.
  * **⛓️ Extensible Pipeline Architecture**: Easily customize and extend the processing pipeline. Swap out models, add new steps, and integrate your own logic.
  * **📝 Natural Language Generation**: Leverage the structured knowledge to generate summaries, answer questions, or create new content.
  * **MIT Licensed**: Open and free for both academic and commercial use.

## Installation

You can install `pumpking` directly from PyPI:

```bash
pip install pumpking
```

## Quick Start

Here's a simple example of how to use the `pumpking` pipeline to process a piece of text, extract its key information, and generate a summary.

```python
from pumpking import PumpkingPipeline

# 1. Initialize the pipeline
# You can configure it with your preferred LLM provider and models.
pipeline = PumpkingPipeline(
    llm_provider='openai',
    api_key='YOUR_API_KEY'
)

# 2. Define your document text
document_text = """
The solar system is a gravitationally bound system of the Sun and the objects that orbit it.
It formed 4.6 billion years ago from the gravitational collapse of a giant interstellar molecular cloud.
The vast majority (99.86%) of the system's mass is in the Sun, with most of the remaining mass
contained in the planet Jupiter. The four inner terrestrial planets—Mercury, Venus, Earth and Mars—are
composed primarily of rock and metal. The four outer giant planets are substantially more massive
than the terrestrials. The two largest, Jupiter and Saturn, are gas giants; the two outermost
planets, Uranus and Neptune, are ice giants.
"""

# 3. Process the document
# This runs the full P.U.M.P.K.I.N.G. pipeline.
processed_doc = pipeline.process(document_text)

# 4. Access the structured knowledge
print("✅ Processing complete!")

# Get the generated summary
print("\n--- Summary ---")
print(processed_doc.summary)

# Access the document chunks
print(f"\n--- Chunks ({len(processed_doc.chunks)}) ---")
for i, chunk in enumerate(processed_doc.chunks):
    print(f"Chunk {i+1}: {chunk.text[:80]}...")

# Access the knowledge graph (conceptual)
print("\n--- Knowledge Graph (Top 3 Triplets) ---")
for i, triplet in enumerate(processed_doc.knowledge_graph.triplets[:3]):
    print(f"- ({triplet.subject}, {triplet.predicate}, {triplet.object})")

```

### Example Output

```text
✅ Processing complete!

--- Summary ---
The solar system, formed 4.6 billion years ago from a molecular cloud, is dominated by the Sun's mass. It consists of four terrestrial inner planets (Mercury, Venus, Earth, Mars) made of rock and metal, and four giant outer planets. The largest, Jupiter and Saturn, are gas giants, while Uranus and Neptune are ice giants.

--- Chunks (3) ---
Chunk 1: The solar system is a gravitationally bound system of the Sun and the objects that...
Chunk 2: The vast majority (99.86%) of the system's mass is in the Sun, with most of th...
Chunk 3: The four outer giant planets are substantially more massive than the terrestrials...

--- Knowledge Graph (Top 3 Triplets) ---
- (solar system, is a, gravitationally bound system)
- (solar system, formed, 4.6 billion years ago)
- (Sun, contains, 99.86% of system's mass)
```

## The P.U.M.P.K.I.N.G. Pipeline Explained

The name `pumpking` stands for our core architecture:

  * **Processing**: Loading raw documents from various sources and preparing them for parsing.
  * **Understanding**: Applying parsing and intelligent chunking to break down documents while preserving semantic meaning.
  * **Modeling**: Transforming the chunks into structured representations like vector embeddings or knowledge graph nodes.
  * **Knowledge Integration**: Connecting the dots—linking entities, normalizing information, and constructing a coherent knowledge base from the document.
  * **Natural-language Generation**: Using the integrated knowledge as a rich context for LLMs to perform downstream tasks like summarization or Q\&A.

## Contributing

Contributions are what make the open-source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

1.  Fork the Project
2.  Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3.  Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4.  Push to the Branch (`git push origin feature/AmazingFeature`)
5.  Open a Pull Request

Please read `CONTRIBUTING.md` for details on our code of conduct and the process for submitting pull requests to us.

## License

This project is licensed under the MIT License. See the [LICENSE](https://www.google.com/search?q=LICENSE) file for details.
