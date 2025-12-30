import os
from typing import List, Optional, Dict, Set, Tuple, Any
from openai import OpenAI
from pydantic import BaseModel, Field

from pumpking.models import NERResult
from pumpking.protocols import (
    NERProviderProtocol,
    SummaryProviderProtocol,
    TopicProviderProtocol,
    ContextualProviderProtocol,
)


class LLMEntityResult(BaseModel):
    """
    Intermediate schema for the LLM output.
    """

    entity: str = Field(..., description="Name of the entity identified.")
    label: str = Field(..., description="Type: PER, ORG, LOC, or MISC.")
    sentences: List[str] = Field(
        ..., description="Exact text of the sentences referring to this entity."
    )


class LLMEntityResponse(BaseModel):
    entities: List[LLMEntityResult]


class LLMTaxonomyResponse(BaseModel):
    """Schema for global taxonomy discovery."""

    topics: List[str] = Field(
        ..., description="Unique topics in the original document language."
    )


class LLMTopicAssignment(BaseModel):
    """Hybrid assignment for validation."""

    block_id: str = Field(..., description="The ID of the block (e.g., 'B1').")
    anchor: str = Field(..., description="The first word of the text in that block.")
    topics: List[str]


class LLMContextAssignment(BaseModel):
    """
    Structured assignment of context to a specific block.
    """

    block_id: str = Field(..., description="The ID of the block (e.g., 'B1').")
    context: str = Field(..., description="The situational context for this block.")
    summary: str = Field(
        ..., description="A concise summary of this block for the next iteration."
    )


class LLMBatchContextResponse(BaseModel):
    """
    Response schema for batch context processing.
    """

    assignments: List[LLMContextAssignment]


class LLMTopicResponse(BaseModel):
    """Structured response for a batch of segments."""

    assignments: List[LLMTopicAssignment]


class LLMProvider(
    NERProviderProtocol,
    SummaryProviderProtocol,
    TopicProviderProtocol,
    ContextualProviderProtocol,
):
    """
    Production-ready LLM Provider with Sliding Window support.
    Delegates semantic grouping to LLM and performs deterministic index mapping.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        default_model: str = "gpt-4o",
        default_temperature: float = 0.0,
    ) -> None:
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.client = None
        if self.api_key:
            self.client = OpenAI(api_key=self.api_key, base_url=base_url)

        self.default_model = default_model
        self.default_temperature = default_temperature

    def _map_sentences_to_indices(
        self, source_sentences: List[str], target_sentences: List[str]
    ) -> List[int]:
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

    def _process_window(
        self, window_sentences: List[str], model: str, temperature: float
    ) -> List[NERResult]:
        if not self.client:
            raise ValueError(
                "OpenAI Client not initialized. Please provide an API Key."
            )

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
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": formatted_input},
            ],
            response_format=LLMEntityResponse,
            temperature=temperature,
        )

        results = []
        if completion.choices[0].message.parsed:
            for item in completion.choices[0].message.parsed.entities:
                local_indices = self._map_sentences_to_indices(
                    window_sentences, item.sentences
                )

                if local_indices:
                    results.append(
                        NERResult(
                            entity=item.entity, label=item.label, indices=local_indices
                        )
                    )
        return results

    def extract_entities(
        self,
        sentences: List[str],
        window_size: int = 20,
        window_overlap: int = 5,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        **kwargs: Any,
    ) -> List[NERResult]:
        if not sentences:
            return []

        if window_overlap >= window_size:
            raise ValueError("window_overlap must be strictly less than window_size")

        active_model = model or self.default_model
        active_temp = (
            temperature if temperature is not None else self.default_temperature
        )

        merged_entities: Dict[Tuple[str, str], Set[int]] = {}

        step = window_size - window_overlap
        step = max(1, step)

        for i in range(0, len(sentences), step):
            window = sentences[i : i + window_size]

            if i > 0 and len(window) < window_overlap and len(sentences) > window_size:
                continue

            window_results = self._process_window(window, active_model, active_temp)

            for res in window_results:
                key = (res.entity, res.label)
                if key not in merged_entities:
                    merged_entities[key] = set()

                global_indices = {local_idx + i for local_idx in res.indices}
                merged_entities[key].update(global_indices)

            if i + window_size >= len(sentences):
                break

        final_results = []
        for (name, label), indices_set in merged_entities.items():
            final_results.append(
                NERResult(entity=name, label=label, indices=sorted(list(indices_set)))
            )

        return final_results

    def summarize(
        self,
        text: str,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        **kwargs: Any,
    ) -> str:
        if not text:
            return ""

        if not self.client:
            raise ValueError(
                "OpenAI Client not initialized. Please provide an API Key."
            )

        active_model = model or self.default_model
        active_temp = (
            temperature if temperature is not None else self.default_temperature
        )

        system_prompt = (
            "You are an expert technical writer specializing in semantic compression. "
            "Summarize the provided text concisely. "
            "CRITICAL INSTRUCTIONS:\n"
            "1. Preserve all key entities (names, organizations, locations, dates).\n"
            "2. Maintain the original semantic meaning and intent.\n"
            "3. Output ONLY the summary text. Do not include introductory phrases like 'Here is a summary'.\n"
            "4. Respond in the SAME LANGUAGE as the input text."
        )

        completion = self.client.chat.completions.create(
            model=active_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text},
            ],
            temperature=active_temp,
        )

        return completion.choices[0].message.content or ""

    def assign_topics(self, chunks: List[str], **kwargs: Any) -> List[List[str]]:
        mode = kwargs.get("topic_mode", "batch_verified")

        # Phase 1: Taxonomy Discovery (Global & Language Consistent)
        taxonomy = self._discover_taxonomy(chunks, **kwargs)

        # Phase 2: Classification
        if mode == "one_by_one":
            return [self._classify_single_chunk(c, taxonomy, **kwargs) for c in chunks]

        return self._classify_with_batch_verification(chunks, taxonomy, **kwargs)

    def _discover_taxonomy(self, chunks: List[str], **kwargs: Any) -> List[str]:
        raw_metadata = []
        batch_size = kwargs.get("taxonomy_batch_size", 15)

        for i in range(0, len(chunks), batch_size):
            batch_text = "\n".join(chunks[i : i + batch_size])
            discovery_prompt = (
                "Extract main themes. CRITICAL: Use the ORIGINAL LANGUAGE of the text. "
                "Do not translate to English."
            )

            response = self.client.chat.completions.create(
                model=kwargs.get("model", self.default_model),
                messages=[
                    {"role": "system", "content": discovery_prompt},
                    {"role": "user", "content": batch_text},
                ],
            )
            raw_metadata.append(response.choices[0].message.content)

        unification_system = (
            "Consolidate these labels into a clean list (max 15 items). "
            "KEEP THE ORIGINAL LANGUAGE (e.g., Spanish). Resolve synonyms."
        )

        completion = self.client.beta.chat.completions.parse(
            model=kwargs.get("model", self.default_model),
            messages=[
                {"role": "system", "content": unification_system},
                {"role": "user", "content": " | ".join(raw_metadata)},
            ],
            response_format=LLMTaxonomyResponse,
        )
        return completion.choices[0].message.parsed.topics

    def _classify_with_batch_verification(
        self, chunks: List[str], taxonomy: List[str], **kwargs: Any
    ) -> List[List[str]]:
        batch_size = kwargs.get("batch_size", 20)
        results = [[] for _ in chunks]

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            marked_text = "\n".join([f"[B{j}] {text}" for j, text in enumerate(batch)])

            prompt = (
                f"Taxonomy: [{', '.join(taxonomy)}]. For each block [Bk], "
                "return the topics and the EXACT first word of its text."
            )

            completion = self.client.beta.chat.completions.parse(
                model=kwargs.get("model", self.default_model),
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": marked_text},
                ],
                response_format=LLMTopicResponse,
            )

            for asm in completion.choices[0].message.parsed.assignments:
                idx_in_batch = int(asm.block_id.replace("B", ""))
                global_idx = i + idx_in_batch

                if global_idx < len(chunks) and chunks[global_idx].lower().startswith(
                    asm.anchor.lower()
                ):
                    results[global_idx] = asm.topics
                else:
                    results[global_idx] = self._classify_single_chunk(
                        chunks[global_idx], taxonomy, **kwargs
                    )
        return results

    def assign_context(self, chunks: List[str], **kwargs: Any) -> List[str]:
        """
        Assigns situational context to each chunk using a rolling memory approach.
        """
        if not chunks:
            return []

        if not self.client:
            raise ValueError(
                "OpenAI Client not initialized. Please provide an API Key."
            )

        active_model = kwargs.get("model", self.default_model)
        active_temp = kwargs.get("temperature", self.default_temperature)

        batch_size = kwargs.get("batch_size", 5)
        results = ["" for _ in chunks]
        previous_summary = "Start of document."

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            marked_text = "\n".join([f"[B{j}] {text}" for j, text in enumerate(batch)])

            system_prompt = (
                "You are an expert technical writer. For each provided text block [Bk], "
                "generate a situational context that anchors it to the document flow. "
                "Use the provided PREVIOUS SUMMARY to maintain continuity. "
                "Also, provide a new summary for the current set of blocks to be used next. "
                "CRITICAL: Respond in the SAME LANGUAGE as the input text."
            )

            user_prompt = (
                f"PREVIOUS SUMMARY: {previous_summary}\n\nBLOCKS:\n{marked_text}"
            )

            completion = self.client.beta.chat.completions.parse(
                model=active_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format=LLMBatchContextResponse,
                temperature=active_temp,
            )

            if completion.choices[0].message.parsed:
                batch_assignments = completion.choices[0].message.parsed.assignments
                for asm in batch_assignments:
                    try:
                        idx_in_batch = int(asm.block_id.replace("B", ""))
                        global_idx = i + idx_in_batch
                        if global_idx < len(chunks):
                            results[global_idx] = asm.context
                    except (ValueError, IndexError):
                        continue

                if batch_assignments:
                    previous_summary = batch_assignments[-1].summary

        return results
