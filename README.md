<p align="center">
<img src="https://github.com/user-attachments/assets/2fd50356-7329-4415-9331-2287e061dad9" width="300"/>
</p>

# Pumpking 🎃

`pumpking` is a Python library for document chunking. It splits a document into a tree of chunks that keeps the lineage of every fragment back to its source, ready to index for RAG (Retrieval-Augmented Generation) or to analyze.

It ships rule-based strategies (regex, paragraphs, sentences, fixed size, sliding windows, adaptive) and semantic strategies that delegate to a model through small protocols (entities, summaries, topics, contextual chunks, zettels). A built-in provider implements those protocols with any OpenAI-compatible API. Strategies compose into pipelines with `>>` (sequence), lists (branches) and `|` (annotations).

## Installation

```bash
pip install pumpking
```

Pumpking needs Python 3.10 or newer.

## Quick Start

This example chunks a document, extracts the named entities of each chunk as an annotation, and summarizes each chunk. Save any Markdown or text file as `report.md` next to the script.

```python
from pathlib import Path

from pumpking.pipeline import PumpkingPipeline, Step, annotate
from pumpking.strategies.advanced import EntityBasedChunking, SummaryChunking
from pumpking.strategies.basic import AdaptiveChunking
from pumpking.strategies.providers import LLMBackend, LLMNERProvider, LLMSummaryProvider

# 1. The backend reads OPENAI_API_KEY and, if set, OPENAI_BASE_URL from the environment.
backend = LLMBackend(default_model="gpt-4o", default_temperature=0.0)

# 2. Providers implement the semantic tasks on top of the backend.
ner_provider = LLMNERProvider(backend=backend)
summary_provider = LLMSummaryProvider(backend=backend)

# 3. Chunk the text and annotate each chunk with its entities, then summarize each chunk.
pipeline = PumpkingPipeline(
    Step(AdaptiveChunking(min_chunk_size=500, max_chunk_size=2000))
    | annotate(EntityBasedChunking(ner_provider=ner_provider), alias="entities")
) >> Step(SummaryChunking(provider=summary_provider))

# 4. Run it on a file (a pathlib.Path, an open file or a plain string).
result = pipeline.run(Path("report.md"))

# 5. Walk the tree: each chunk node holds its chunk and, below it, the summary node.
for chunk_node in result.branches:
    chunk = chunk_node.results[0]
    entities = sorted({e.entity for e in chunk.annotations.get("entities", [])})
    print(f"\n--- Chunk {chunk.id} ---")
    print("Text:", chunk.content[:80], "...")
    print("Entities:", ", ".join(entities) or "none")
    for summary_node in chunk_node.branches:
        for summary in summary_node.results:
            print("Summary:", summary.content)
```

Run it with your key in the environment:

```bash
export OPENAI_API_KEY=sk-...
python quickstart.py
```

To use a local model instead, point the backend at any server that speaks the OpenAI chat completions API with structured outputs, such as Ollama or vLLM. Local servers that do not check keys accept any value:

```bash
export OPENAI_BASE_URL=http://localhost:11434/v1
export OPENAI_API_KEY=unused
```

and pass a model that server knows, for example `LLMBackend(default_model="llama3.1")`.

If the endpoint cannot be reached, the first call raises `pumpking.exceptions.LLMBackendError` with the URL it tried.

More scripts are in [`examples/`](examples/). They run without network access or a key.

---

## Strategies

Pumpking distinguishes **basic strategies** (rules and structure) from **advanced strategies** (document structure or a semantic provider).

### Basic strategies

* **`FixedSizeChunking`**: Splits text into windows of a fixed character length, with optional overlap.
* **`RegexChunking`**: Splits text on any regular expression.
* **`ParagraphChunking`**: Splits on blank lines (`\n\n`), treating paragraphs as atomic units.
* **`SentenceChunking`**: Splits on sentence punctuation (`.`, `?`, `!`) with look-behind rules.
* **`SlidingWindowChunking`**: Builds overlapping windows of **words**.
* **`AdaptiveChunking`**: Accumulates whole sentences into chunks between `min_chunk_size` and `max_chunk_size` characters, never breaking a sentence.

### Advanced strategies

* **`HierarchicalChunking`**: Parses Markdown headers (`H1` -> `H2`) into a nested tree of sections, and can apply other strategies to each section's body.
* **`EntityBasedChunking`** (`ner_provider=`): Finds named entities (people, organizations, places) and links each one back to the sentences that mention it.
* **`SummaryChunking`** (`provider=`): Splits text into blocks and summarizes each block, keeping the source as the summary's child.
* **`TopicBasedChunking`** (`topic_provider=`): Discovers a topic taxonomy and groups chunks under the topics they belong to.
* **`ContextualChunking`** (`provider=`): Adds document-level context to each fragment (who "he" is, which section it comes from), as in contextual retrieval.
* **`ZettelkastenChunking`** (`zettel_provider=`): Extracts atomic ideas (zettels) linked to the fragments that support them and to each other.

---

## Providers

The advanced strategies depend on protocols defined in `pumpking.protocols`, not on a model vendor:

* **`NERProviderProtocol`**
* **`SummaryProviderProtocol`**
* **`TopicProviderProtocol`**
* **`ContextualProviderProtocol`**
* **`ZettelProviderProtocol`**

Any class with the protocol's method works with the matching strategy, so you can back a strategy with spaCy, a local seq2seq model or a lookup table. [`examples/02_custom_provider.py`](examples/02_custom_provider.py) shows one.

### Built-in LLM providers

Pumpking includes one implementation of each protocol on top of a shared **`LLMBackend`**, which talks to any OpenAI-compatible API:

```python
from pumpking.strategies.providers import LLMBackend

backend = LLMBackend(
    base_url=None,           # default: OPENAI_BASE_URL, else https://api.openai.com/v1
    api_key=None,            # default: OPENAI_API_KEY
    default_model="gpt-4o",
    default_temperature=0.0,
    timeout=60.0,            # seconds per request
    max_retries=2,           # retries per request before raising
)
```

Inject the backend into the providers:

* **`LLMNERProvider`**: Extracts entities and resolves coreferences.
* **`LLMSummaryProvider`**: Summarizes chunks, five per request.
* **`LLMTopicProvider`**: Discovers a taxonomy and assigns topics.
* **`LLMContextualProvider`**: Generates situational context for each fragment.
* **`LLMZettelProvider`**: Synthesizes atomic ideas (zettels).

```python
from pumpking.strategies.providers import LLMNERProvider, LLMZettelProvider

ner_provider = LLMNERProvider(backend=backend)
zettel_provider = LLMZettelProvider(backend=backend)
```

### Performance: LLM calls are serial

Every provider call is a blocking request, and requests run one after another. `SummaryChunking` sends one request per five blocks, `EntityBasedChunking` one per window of sentences, and pipeline branches run in sequence. A long document therefore takes as many round trips as it has batches: test on short texts first, and use `timeout` and `max_retries` to bound how long a slow endpoint can stall a run.

### Security: API keys

Pass the key through the `OPENAI_API_KEY` environment variable (or a secret manager that sets it), not as a literal in code you commit. `repr(backend)` shows the endpoint and model but not the key; avoid logging `backend.api_key` or `backend.client`.

---

## Pipelines

`PumpkingPipeline` runs steps over a tree. Each step's output chunks become nodes below the chunks they came from.

### Sequential execution

The output of one step is the input of the next. Chain steps with `>>`.

```python
pipeline = PumpkingPipeline(
    Step(AdaptiveChunking(1000, 3000))
) >> Step(SummaryChunking(provider=summary_provider))
```

### Branches

Pass a **list of steps** to run several strategies on the same input. Each branch adds its own nodes.

```python
pipeline = PumpkingPipeline(
    Step(AdaptiveChunking(1000, 3000))
) >> [
    Step(TopicBasedChunking(topic_provider=topic_provider)),
    Step(EntityBasedChunking(ner_provider=ner_provider)),
]
```

### Annotations

An **annotator** runs on each chunk a step produces and stores its result in `chunk.annotations` under an alias, without adding nodes. Attach it with `|` and `annotate`.

```python
step = Step(AdaptiveChunking(1000, 3000)) | annotate(
    EntityBasedChunking(ner_provider=ner_provider), alias="entities"
)
```

---

## Contributing

Contributions are welcome. Read [`CONTRIBUTING.md`](CONTRIBUTING.md) for the design rules and the development setup, then open a pull request.

## Developed By

**Pumpking** is a collaborative open-source project co-developed by:

* **[GIA-UH](https://github.com/gia-uh/)** (Grupo de Inteligencia Artificial - Universidad de La Habana)
* **[Syalia S.R.L.](https://syalia.com/)**

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
