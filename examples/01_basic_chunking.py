"""
Rule-based chunking: no model, no network.

Run: python examples/01_basic_chunking.py
"""

from pumpking.pipeline import PumpkingPipeline, Step
from pumpking.strategies.advanced import HierarchicalChunking
from pumpking.strategies.basic import (
    AdaptiveChunking,
    ParagraphChunking,
    SentenceChunking,
    SlidingWindowChunking,
)

DOCUMENT = """# Solar energy

Solar panels turn sunlight into electricity. Their output depends on the angle of the sun.
A panel facing south collects more energy in the northern hemisphere.

## Storage

Batteries store the surplus produced at noon. They release it in the evening, when demand peaks.

## Costs

The price of panels fell by more than eighty percent in ten years. Installation now costs more
than the panels themselves.
"""


def main() -> None:
    # Each strategy can run on its own, on a string.
    for strategy in [
        ParagraphChunking(),
        SentenceChunking(),
        SlidingWindowChunking(window_size=12, overlap=4),
        AdaptiveChunking(min_chunk_size=80, max_chunk_size=200),
    ]:
        chunks = PumpkingPipeline(Step(strategy)).run(DOCUMENT).branches
        print(f"{type(strategy).__name__}: {len(chunks)} chunks")
        print("  first:", chunks[0].results[0].content[:70])

    # HierarchicalChunking follows the Markdown headers and splits each section's body.
    root = PumpkingPipeline(
        Step(HierarchicalChunking(strategies=[SentenceChunking()]))
    ).run(DOCUMENT)

    def show(payload, depth=0):
        print("  " * depth + "- " + (payload.content or "")[:60].replace("\n", " "))
        for child in payload.children:
            show(child, depth + 1)

    print("\nHierarchicalChunking tree:")
    for node in root.branches:
        for payload in node.results:
            show(payload)


if __name__ == "__main__":
    main()
