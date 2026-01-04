# Contributing to Pumpking 🎃

First off, thank you for considering contributing to **Pumpking**! It's people like you that make the open-source community such an amazing place to learn, inspire, and create.

Pumpking is a co-development effort between **[GIA-UH](https://github.com/gia-uh/)** and **[Syalia S.R.L.](https://github.com/syalia-srl)**, and we welcome contributions from everyone.

## 🧠 The "Pumpking Way"

Before you write code, please understand our core design philosophy. We prioritize **architecture over convenience**:

1.  **Protocol-First**: We never couple strategies directly to implementations (e.g., OpenAI). Always define a `Protocol` in `src/pumpking/protocols.py` first.
2.  **Strict Typing**: We use Python's type system aggressively. `Any` is discouraged. If it's a list of chunks, it's `List[ChunkPayload]`, not `list`.
3.  **Dependency Injection**: Strategies should receive their dependencies (Providers) in `__init__`. Do not hardcode API calls inside a strategy.
4.  **Fluent Interface**: Our Pipeline is a DAG. Ensure your steps play nicely with the `>>` and `|` operators.

---

## 🛠️ Development Setup

We use **[uv](https://github.com/astral-sh/uv)** for ultra-fast dependency management and virtual environments.

### 1. Prerequisites
* Python 3.9+
* `uv` installed (`pip install uv` or `curl -LsSf https://astral.sh/uv/install.sh | sh`)

### 2. Installation
Clone the repository and sync the environment:

```bash
git clone [https://github.com/gia-uh/pumpking.git](https://github.com/gia-uh/pumpking.git)
cd pumpking

# Create virtualenv and install dependencies (including dev)
uv sync

```

### 3. Running Tests

We use `pytest` for our test suite. Ensure all tests pass before submitting a PR.

```bash
# Run all tests
uv run pytest

# Run a specific test file
uv run pytest tests/test_pipeline.py

```

---

## 🚀 How to Contribute

### 1. Adding a New Strategy

If you want to add a new way to chunk or process documents:

1. **Check Protocols**: Does it need an external provider (like an LLM)? If so, define/reuse a Protocol in `src/pumpking/protocols.py`.
2. **Implement Strategy**: Create your class in `src/pumpking/strategies/` (basic or advanced).
* Inherit from `BaseStrategy`.
* Implement `execute(self, input_data, context)`.


3. **Add Tests**: Create a corresponding `tests/test_your_strategy.py`.

### 2. Adding a New Provider

If you want to add support for a new backend (e.g., Anthropic, Ollama, HuggingFace):

1. **Do NOT modify the Strategy**: The strategy shouldn't know about the backend.
2. **Implement the Provider**: Go to `src/pumpking/strategies/providers.py` and implement the relevant Protocol (e.g., `NERProviderProtocol`).
3. **Update the Backend**: If necessary, update `LLMBackend` to support the new client authentication/connection.

---

## 📏 Style Guidelines

* **Code Style**: We follow standard PEP 8.
* **Docstrings**: Every class and public method must have a docstring (Google Style).
* **Type Hints**: Mandatory for all function arguments and return values.

```python
# GOOD
def process(self, text: str) -> List[ChunkPayload]:
    """Processes text and returns chunks."""
    ...

# BAD
def process(self, text):
    ...

```

---

## workflow

1. **Fork** the repo on GitHub.
2. **Clone** the project to your own machine.
3. **Commit** changes to your own branch.
4. **Push** your work back up to your fork.
5. Submit a **Pull Request** so that we can review your changes.

NOTE: Be sure to merge the latest from "upstream" before making a pull request!

## 📜 Code of Conduct

Please note that this project is released with a Contributor Code of Conduct. By participating in this project you agree to abide by its terms.

Happy Hacking! 🎃



