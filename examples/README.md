# Examples

Each script runs from the repository root with `python examples/<script>.py` after `pip install pumpking` (or `uv run python examples/<script>.py` in a clone). None needs network access or an API key; CI runs all of them.

| Script | What it shows |
|---|---|
| `01_basic_chunking.py` | Rule-based strategies and `HierarchicalChunking` on a Markdown document. |
| `02_custom_provider.py` | Your own NER and summary providers in plain Python, with an annotation and a sequential step. |
| `03_llm_pipeline.py` | All five built-in LLM providers. Uses the real API when `OPENAI_API_KEY` is set, and the fake server otherwise. |
| `fake_openai.py` | A fake OpenAI chat completions server for offline runs: `python examples/fake_openai.py --port 8765`. |
