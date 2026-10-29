"""
The built-in LLM providers on an OpenAI-compatible API.

With OPENAI_API_KEY set (and OPENAI_BASE_URL for a local server), this calls the real model.
Without a key it starts the fake server in fake_openai.py, so the script runs offline; the
fake echoes the input, so its "summaries" and "entities" are placeholders.

Run: python examples/03_llm_pipeline.py
"""

import os

from pumpking.pipeline import PumpkingPipeline, Step, annotate
from pumpking.strategies.advanced import (
    ContextualChunking,
    EntityBasedChunking,
    SummaryChunking,
    TopicBasedChunking,
    ZettelkastenChunking,
)
from pumpking.strategies.basic import AdaptiveChunking
from pumpking.strategies.providers import (
    LLMBackend,
    LLMContextualProvider,
    LLMNERProvider,
    LLMSummaryProvider,
    LLMTopicProvider,
    LLMZettelProvider,
)

DOCUMENT = """Marie Curie moved to Paris in 1891 to study physics at the Sorbonne. She shared the
1903 Nobel Prize in Physics for research on radioactivity.

In 1911 she received a second Nobel Prize, this time in chemistry, for discovering polonium
and radium. She was the first person to win Nobel Prizes in two sciences.

During the First World War she organized mobile X-ray units for field hospitals. Her daughter
Irene worked with her on these units."""


def make_backend() -> LLMBackend:
    if os.getenv("OPENAI_API_KEY"):
        return LLMBackend(default_model=os.getenv("PUMPKING_MODEL", "gpt-4o-mini"))
    from fake_openai import serve

    _, url = serve()
    print(f"OPENAI_API_KEY is not set: using the fake server at {url}\n")
    return LLMBackend(base_url=url, api_key="unused", default_model="fake")


def main() -> None:
    backend = make_backend()

    pipeline = PumpkingPipeline(
        Step(AdaptiveChunking(min_chunk_size=100, max_chunk_size=400))
        | annotate(EntityBasedChunking(ner_provider=LLMNERProvider(backend)), alias="entities")
    ) >> [
        Step(SummaryChunking(provider=LLMSummaryProvider(backend)), alias="summary"),
        Step(ContextualChunking(provider=LLMContextualProvider(backend)), alias="context"),
    ]

    root = pipeline.run(DOCUMENT)
    for chunk_node in root.branches:
        chunk = chunk_node.results[0]
        print("Chunk:", chunk.content[:70], "...")
        print("  entities:", sorted({e.entity[:40] for e in chunk.annotations["entities"]}))
        for branch in chunk_node.branches:
            for result in branch.results:
                print(f"  {branch.strategy_label}:", (result.content or "")[:70])

    # Whole-document strategies work on the text directly.
    topics = PumpkingPipeline(
        Step(TopicBasedChunking(topic_provider=LLMTopicProvider(backend)))
    ).run(DOCUMENT)
    print("\nTopics:", [r.topic[:40] for n in topics.branches for r in n.results])

    zettels = PumpkingPipeline(
        Step(ZettelkastenChunking(zettel_provider=LLMZettelProvider(backend)))
    ).run(DOCUMENT)
    print("Zettels:", [r.hypothesis[:40] for n in zettels.branches for r in n.results])


if __name__ == "__main__":
    main()
