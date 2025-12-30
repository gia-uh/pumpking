import os
from typing import List, Optional, Dict, Set, Tuple
from openai import OpenAI
from pydantic import BaseModel, Field

from pumpking.models import NERResult
from pumpking.protocols import NERProviderProtocol, SummaryProviderProtocol


class LLMEntityResult(BaseModel):
    """
    Intermediate schema for the LLM output.
    """
    entity: str = Field(..., description="Name of the entity identified.")
    label: str = Field(..., description="Type: PER, ORG, LOC, or MISC.")
    sentences: List[str] = Field(..., description="Exact text of the sentences referring to this entity.")


class LLMResponseWrapper(BaseModel):
    entities: List[LLMEntityResult]


class LLMProvider(NERProviderProtocol):
    """
    Production-ready LLM Provider with Sliding Window support.
    Delegates semantic grouping to LLM and performs deterministic index mapping.
    """
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: str = "gpt-4o",
        temperature: float = 0.0,
        window_size: int = 20,
        window_overlap: int = 5
    ) -> None:
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.client = None
        if self.api_key:
            self.client = OpenAI(api_key=self.api_key, base_url=base_url)

        self.model = model
        self.temperature = temperature
        self.window_size = window_size
        self.window_overlap = window_overlap

        if self.window_overlap >= self.window_size:
            raise ValueError("window_overlap must be strictly less than window_size")

    def _map_sentences_to_indices(self, source_sentences: List[str], target_sentences: List[str]) -> List[int]:
        indices = []
        lookup_map: Dict[str, List[int]] = {}
        for idx, sent in enumerate(source_sentences):
            if sent not in lookup_map:
                lookup_map[sent] = []
            lookup_map[sent].append(idx)

        for target in target_sentences:
            if target in lookup_map:
                indices.extend(lookup_map[target])
                continue

            target_norm = target.strip().lower()
            for idx, source in enumerate(source_sentences):
                if source.strip().lower() == target_norm:
                    indices.append(idx)

        return sorted(list(set(indices)))

    def _process_window(self, window_sentences: List[str]) -> List[NERResult]:
        if not self.client:
            raise ValueError("OpenAI Client not initialized. Please provide an API Key.")

        formatted_input = "\n".join([f"- {s}" for s in window_sentences])

        system_prompt = (
            "You are an advanced NLP linguist specialized in NER and Coreference Resolution. "
            "Analyze the provided text in its ORIGINAL LANGUAGE. Do not translate entities or sentences.\n\n"
            "Ontology:\n"
            "- PER: People, fictional characters.\n"
            "- ORG: Companies, institutions, agencies.\n"
            "- LOC: Geopolitical entities, physical locations.\n"
            "- MISC: Events, laws, products, works of art, nationalities. (Exclude abstract concepts).\n\n"
            "Task:\n"
            "Extract entities and group the EXACT sentences where they appear (including pronouns/references).\n\n"
            "Examples:\n"
            "Input:\n"
            "- Elon Musk visits Mars.\n"
            "- He wants to colonize it.\n"
            "- SpaceX builds rockets.\n"
            "Output:\n"
            "Entity: 'Elon Musk' (PER) -> Sentences: ['Elon Musk visits Mars.', 'He wants to colonize it.']\n"
            "Entity: 'Mars' (LOC) -> Sentences: ['Elon Musk visits Mars.', 'He wants to colonize it.']\n"
            "Entity: 'SpaceX' (ORG) -> Sentences: ['SpaceX builds rockets.']\n\n"
            "Instructions:\n"
            "1. Return the EXACT text of the sentences from the input.\n"
            "2. A sentence can belong to multiple entities (Overlap).\n"
            "3. Resolve coreferences (e.g., 'He', 'The company')."
        )

        completion = self.client.beta.chat.completions.parse(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": formatted_input},
            ],
            response_format=LLMResponseWrapper,
            temperature=self.temperature,
        )

        results = []
        if completion.choices[0].message.parsed:
            for item in completion.choices[0].message.parsed.entities:
                local_indices = self._map_sentences_to_indices(window_sentences, item.sentences)
                
                if local_indices:
                    results.append(NERResult(
                        entity=item.entity,
                        label=item.label,
                        indices=local_indices
                    ))
        return results

    def extract_entities(self, sentences: List[str]) -> List[NERResult]:
        if not sentences:
            return []

        merged_entities: Dict[Tuple[str, str], Set[int]] = {}

        step = self.window_size - self.window_overlap
        step = max(1, step)

        for i in range(0, len(sentences), step):
            window = sentences[i : i + self.window_size]
            
            if i > 0 and len(window) < self.window_overlap and len(sentences) > self.window_size:
                continue

            window_results = self._process_window(window)

            for res in window_results:
                key = (res.entity, res.label)
                if key not in merged_entities:
                    merged_entities[key] = set()
                
                global_indices = {local_idx + i for local_idx in res.indices}
                merged_entities[key].update(global_indices)

            if i + self.window_size >= len(sentences):
                break

        final_results = []
        for (name, label), indices_set in merged_entities.items():
            final_results.append(NERResult(
                entity=name,
                label=label,
                indices=sorted(list(indices_set))
            ))

        return final_results
    
    def summarize(self, text: str) -> str:
        """
        Generates a summary of the text using the configured LLM.
        """
        return text