<p align="center">
<img src="https://github.com/user-attachments/assets/2fd50356-7329-4415-9331-2287e061dad9" width="300"/>
</p>

# Pumpking 🎃

`pumpking` is an open-source Python library designed to streamline the complex process of document chunking, parsing, and representation. It provides a flexible and powerful pipeline to transform unstructured documents into structured, queryable knowledge.

Whether you're building a RAG (Retrieval-Augmented Generation) system, a document analysis tool, or a knowledge extraction service, `pumpking` provides the foundational blocks to get you there faster.

## The P.U.M.P.K.I.N.G. Pipeline Explained

The name `pumpking` stands for our core architecture:

* **Processing**: Loading raw documents from various sources and preparing them for parsing.
* **Understanding**: Applying parsing and intelligent chunking to break down documents while preserving semantic meaning.
* **Modeling**: Transforming the chunks into structured representations like vector embeddings or knowledge graph nodes.
* **Knowledge Integration**: Connecting the dots—linking entities, normalizing information, and constructing a coherent knowledge base from the document.
* **Natural-language Generation**: Using the integrated knowledge as a rich context for LLMs to perform downstream tasks like summarization or Q&A.

---

## 🏗️ Core Philosophy: Protocols over Implementations

Pumpking is built on a **Protocol-First Architecture**. We define strict interfaces (Python Protocols) for semantic tasks, decoupling the *strategy* from the *implementation*.

This means you are **not locked into Large Language Models (LLMs)**. While we provide LLM-based implementations out of the box for convenience, you can easily implement these protocols using other paradigms:

* **`NERProviderProtocol`**: Could be implemented with **SpaCy**, **GluonNLP**, or a **Regex** lookup table.
* **`TopicProviderProtocol`**: Could use **BERTopic** or **LDA**.
* **`SummaryProviderProtocol`**: Could use a standard **T5** or **BART** model locally.

As long as your class adheres to the protocol defined in `pumpking.protocols`, it will work seamlessly with any Advanced Strategy.

---

## 🛠️ Strategies

Pumpking distinguishes between **Basic Strategies** (algorithmic/heuristic) and **Advanced Strategies** (semantic/inference-based).

### Basic Strategies

Foundational splitting logic relying on rules, regex, and structure.

* **`FixedSizeChunking`**: Rigidly splits text into windows of a specific character length with optional overlap.
* **`RegexChunking`**: A versatile splitter that divides text based on any configurable regular expression pattern.
* **`ParagraphChunking`**: Semantic splitting based on double newlines (`\n\n`), treating paragraphs as atomic units.
* **`SentenceChunking`**: Uses linguistic heuristics (punctuation look-behinds like `.` `?` `!`) to split text into grammatical sentences.
* **`SlidingWindowChunking`**: Operates on **words** (splitting by whitespace) to create overlapping windows, maintaining context flow better than character splitting.
* **`AdaptiveChunking`**: A smart accumulation strategy. It builds chunks sentence-by-sentence up to a `max_chunk_size` limit, ensuring sentences are never broken and context is maximized.

### Advanced Strategies

These strategies perform deep semantic analysis by delegating work to an injected **Provider**.

* **`HierarchicalChunking`**: Respects document structure by parsing Markdown headers (`H1` -> `H2`). It builds a nested tree where sections contain their subsections.
* **`EntityBasedChunking`**: Decomposes text to identify named entities (People, Organizations). It preserves strict lineage, linking every extracted entity back to the specific source fragments.
* **`SummaryChunking`**: Splits text into semantic blocks and generates concise summaries for each.
* **`TopicBasedChunking`**: Analyzes content to identify latent topics or themes, grouping content clusters under shared semantic topics.
* **`ContextualChunking`**: Designed for RAG. It enriches individual text fragments with "Global Document Context"—resolving pronouns and adding missing background info.
* **`ZettelkastenChunking`**: Transforms text into **Zettels**—atomic units of knowledge or hypotheses linked to evidence.

---

## 🤖 Built-in LLM Providers

For immediate productivity, Pumpking includes a reference implementation of the protocols powered by **LLMs**. These are production-ready and handle the complexity of structured output parsing and API communication.

They use a decoupled architecture with a shared **`LLMBackend`**.

### 1. The Backend

Manages the connection to the LLM. You must configure the API key (or set `OPENAI_API_KEY`) and the default model.

```python
from pumpking.strategies.providers import LLMBackend

# Configuration based on source code signature:
backend = LLMBackend(
    api_key="sk-...",           # Your API Key (optional if env var set)
    default_model="gpt-4o",     # Model identifier
    default_temperature=0.0,    # Controls randomness
    base_url=None               # Optional: Set for local LLMs
)

```

### 2. The Providers

Inject the `backend` into any of the available providers:

* **`LLMNERProvider`**: Extracts entities and resolves coreferences.
* **`LLMSummaryProvider`**: Generates summaries for chunks.
* **`LLMTopicProvider`**: Discovers taxonomies and assigns topics.
* **`LLMContextualProvider`**: Generates situational context (Contextual RAG).
* **`LLMZettelProvider`**: Synthesizes atomic hypotheses (Zettels).

```python
from pumpking.strategies.providers import LLMNERProvider, LLMZettelProvider

ner_provider = LLMNERProvider(backend=backend)
zettel_provider = LLMZettelProvider(backend=backend)

```

---

## 🚀 The Pipeline Architecture

Pumpking's `Pipeline` implements a Directed Acyclic Graph (DAG) executor. It is designed to be composable, allowing you to chain strategies, run them in parallel, or attach them as metadata extractors.

### 1. Sequential Execution (Chaining)

The most common pattern. The output of `Step A` becomes the input of `Step B`.
Use the `>>` (Shift) operator to chain steps.

```python
# Pattern: Input -> Chunk -> Summarize -> Output
pipeline = PumpkingPipeline(
    Step(AdaptiveChunking(1000, 3000))
) >> Step(SummaryChunking(provider=summary_provider))

```

### 2. Parallel Execution (Branching)

You can split the processing flow into multiple branches that run on the same input data. The results are merged back into the stream for the next step.
Pass a **list of Steps** to create branches.

```python
# Pattern: Input -> Chunk -> [Identify Topics, Extract Entities] -> Merge Results
pipeline = PumpkingPipeline(
    Step(AdaptiveChunking(1000, 3000))
) >> [
    Step(TopicBasedChunking(provider=topic_provider)), 
    Step(EntityBasedChunking(provider=ner_provider))
]

```

### 3. Annotations (Enrichment)

Sometimes you want to extract metadata *about* a chunk without transforming the chunk itself. An **Annotator** runs "inside" a step, attaching its result to the `chunk.annotations` dictionary.
Use the `|` (Pipe) operator and the `annotate` helper.

```python
# Pattern: Chunk (and tag each chunk with Sentiment)
step = Step(AdaptiveChunking(1000, 3000)) | annotate(SentimentStrategy(), alias="sentiment")

```
---

## Quick Start: A Real-World Example

This example demonstrates a complete pipeline using the **built-in LLM providers**. It chunks a document, extracts entities as metadata (Annotation), and summarizes the chunks (Sequential).

```python
import os
from pumpking.pipeline import PumpkingPipeline, Step, annotate
from pumpking.strategies.basic import AdaptiveChunking
from pumpking.strategies.advanced import EntityBasedChunking, SummaryChunking
from pumpking.strategies.providers import LLMBackend, LLMNERProvider, LLMSummaryProvider

# 1. Initialize the Backend
backend = LLMBackend(
    default_model="gpt-4o",
    default_temperature=0.0
)

# 2. Initialize Providers
ner_provider = LLMNERProvider(backend=backend)
summary_provider = LLMSummaryProvider(backend=backend)

# 3. Define the Pipeline using Fluent Syntax
# Flow:
#  1. Chunk text (Adaptive)
#     └── AND Identify Entities (Annotation: 'entities')
#  2. THEN Summarize those chunks (Sequential)

pipeline = PumpkingPipeline(
    Step(AdaptiveChunking(min_chunk_size=500, max_chunk_size=2000)) 
    | annotate(EntityBasedChunking(ner_provider=ner_provider), alias="entities")
) >> Step(SummaryChunking(provider=summary_provider))

# 4. Run on a file (Pumpking handles I/O automatically)
result_root = pipeline.run(Path("docs/annual_report.md"))

# 5. Inspect the Graph
for node in result_root.branches:
    for chunk in node.results:
        print(f"\n--- Chunk {chunk.id} ---")
        print(f"Summary: {chunk.content[:100]}...")
        
        # Access original content (preserved in children by SummaryChunking)
        if chunk.children:
             print(f"Original Text: {chunk.children[0].content[:50]}...")

        # Access Annotations (passed down or stored on children depending on strategy)
        # Note: SummaryChunking creates new nodes, so annotations from Step 1 
        # are typically found on the source chunks (children).
        source_chunk = chunk.children[0]
        if "entities" in source_chunk.annotations:
            print("Entities in original text:", len(source_chunk.annotations["entities"]))

```

---

## Contributing

Contributions are what make the open-source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

Please read `CONTRIBUTING.md` for details on our code of conduct and the process for submitting pull requests to us.

---

## Developed By

**Pumpking** is a collaborative open-source project co-developed by:

* **[GIA-UH](https://github.com/gia-uh/)** (Grupo de Inteligencia Artificial - Universidad de La Habana)
* **[Syalia S.R.L.](https://syalia.com/)**

---

## License

This project is licensed under the MIT License. See the [LICENSE](https://www.google.com/search?q=LICENSE) file for details.