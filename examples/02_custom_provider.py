"""
Advanced strategies with your own providers: no model, no network.

The advanced strategies depend on protocols, not on an LLM. Here two small pure-Python
classes implement NERProviderProtocol and SummaryProviderProtocol, and a pipeline uses them
for an annotation and a branch.

Run: python examples/02_custom_provider.py
"""

import re
from typing import Any, Dict, List

from pumpking.models import ChunkPayload, EntityChunkPayload
from pumpking.pipeline import PumpkingPipeline, Step, annotate
from pumpking.protocols import NERProviderProtocol, SummaryProviderProtocol
from pumpking.strategies.advanced import EntityBasedChunking, SummaryChunking
from pumpking.strategies.basic import ParagraphChunking

DOCUMENT = """Ada Lovelace worked with Charles Babbage in London. She wrote the first published
algorithm for the Analytical Engine.

Babbage designed the Difference Engine before the Analytical Engine. London institutions
funded part of his work, but the machine was never finished."""

GAZETTEER = {
    "Ada Lovelace": "PER",
    "Charles Babbage": "PER",
    "Babbage": "PER",
    "London": "LOC",
    "Analytical Engine": "MISC",
    "Difference Engine": "MISC",
}


class GazetteerNER(NERProviderProtocol):
    """Finds the names of a fixed list in each sentence."""

    def extract_entities(
        self, chunks: List[ChunkPayload], **kwargs: Any
    ) -> List[EntityChunkPayload]:
        found: Dict[str, List[ChunkPayload]] = {}
        for chunk in chunks:
            for name in GAZETTEER:
                if re.search(rf"\b{re.escape(name)}\b", chunk.content or ""):
                    found.setdefault(name, []).append(chunk)
        return [
            EntityChunkPayload(entity=name, type=GAZETTEER[name], children=sentences)
            for name, sentences in found.items()
        ]


class FirstSentenceSummary(SummaryProviderProtocol):
    """Summarizes a chunk as its first sentence."""

    def summarize(self, chunks: List[ChunkPayload], **kwargs: Any) -> List[ChunkPayload]:
        return [
            ChunkPayload(content=re.split(r"(?<=\.)\s", c.content or "")[0], children=[c])
            for c in chunks
        ]


def main() -> None:
    pipeline = PumpkingPipeline(
        Step(ParagraphChunking())
        | annotate(EntityBasedChunking(ner_provider=GazetteerNER()), alias="entities")
    ) >> Step(SummaryChunking(provider=FirstSentenceSummary()))

    root = pipeline.run(DOCUMENT)

    for i, paragraph_node in enumerate(root.branches, 1):
        paragraph = paragraph_node.results[0]
        entities = [f"{e.entity} ({e.type})" for e in paragraph.annotations["entities"]]
        print(f"Paragraph {i}: {', '.join(entities)}")
        for summary_node in paragraph_node.branches:
            for summary in summary_node.results:
                print(f"  summary: {summary.content}")


if __name__ == "__main__":
    main()
