import os
import difflib
import uuid
from typing import List, Optional, Dict, Set, Tuple, Any
from openai import OpenAI
from pydantic import BaseModel, Field

from pumpking.models import (
    ChunkPayload,
    ZettelChunkPayload,
    EntityChunkPayload,
    TopicChunkPayload,
    ContextualChunkPayload
)
from pumpking.protocols import (
    NERProviderProtocol,
    SummaryProviderProtocol,
    TopicProviderProtocol,
    ContextualProviderProtocol,
    ZettelProviderProtocol,
)


class LLMEntityResult(BaseModel):
    """
    Internal Data Transfer Object for a single entity detected by the LLM.
    """

    entity: str = Field(..., description="Name of the entity identified.")
    label: str = Field(..., description="Type: PER, ORG, LOC, or MISC.")
    sentences: List[str] = Field(
        ..., description="Exact text of the sentences referring to this entity."
    )


class LLMEntityResponse(BaseModel):
    """
    Internal container for the structured response from the LLM.
    """

    entities: List[LLMEntityResult]
class LLMTopicAssignment(BaseModel):
    """
    Represents the assignment of semantic topics to a specific text block.
    Uses anchoring to map back to source chunks deterministically.
    """
    anchor: str = Field(..., description="The first 10 to 15 words of the text block exactly as they appear.")
    topics: List[str] = Field(..., description="The list of applicable topics derived from the provided taxonomy.")

class LLMTopicResponse(BaseModel):
    assignments: List[LLMTopicAssignment]

class LLMTaxonomyResponse(BaseModel):
    topics: List[str]

class LLMContextAssignment(BaseModel):
    """
    Data structure for mapping generated context to source fragments via content echoing.
    """

    quote: str = Field(
        ..., description="A short excerpt from the start of the fragment"
    )
    context: str = Field(..., description="The situational anchoring information")


class LLMContextResponse(BaseModel):
    """
    Container for batch-processed contextual assignments.
    """

    assignments: List[LLMContextAssignment]


class LLMTopicResponse(BaseModel):
    """Structured response for a batch of segments."""

    assignments: List[LLMTopicAssignment]


class LLMZettelRef(BaseModel):
    """
    Data Transfer Object representing a single Zettel extracted by the LLM.
    """

    concept_handle: str = Field(
        ...,
        description="A short, unique, and descriptive title (2-5 words) identifying this specific concept. This handle acts as a semantic anchor for linking.",
    )
    hypothesis: str = Field(
        ...,
        description="The complete, atomic idea or thesis statement. It must be self-contained and understood without reading the original source text.",
    )
    tags: List[str] = Field(
        default_factory=list, description="Taxonomic keywords for categorization."
    )
    direct_quotes: List[str] = Field(
        ...,
        description="Exact text fragments copied verbatim from the provided input that support this hypothesis. Used to trace back to source chunks.",
    )
    related_concept_handles: List[str] = Field(
        default_factory=list,
        description="A list of 'concept_handles' representing semantic relationships. Can include handles generated in the current batch or provided in the previous context.",
    )


class LLMZettelResponse(BaseModel):
    """
    Container for the list of Zettels extracted in a single LLM inference call.
    """

    zettels: List[LLMZettelRef]


class LLMProvider(
    NERProviderProtocol,
    SummaryProviderProtocol,
    TopicProviderProtocol,
    ContextualProviderProtocol,
    ZettelProviderProtocol,
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
        """
        Identifies the indices of sentences in the source list that match the target sentences.
        Uses exact matching first, falling back to fuzzy matching for robustness against
        minor LLM hallucinations or formatting changes.
        """
        indices = set()

        lookup_map: Dict[str, List[int]] = {}
        for idx, sent in enumerate(source_sentences):
            if sent not in lookup_map:
                lookup_map[sent] = []
            lookup_map[sent].append(idx)

        source_norm = [(s.strip().lower(), i) for i, s in enumerate(source_sentences)]

        for target in target_sentences:
            if target in lookup_map:
                indices.update(lookup_map[target])
                continue

            target_clean = target.strip().lower()

            found_normalized = False
            for source_clean, idx in source_norm:
                if source_clean == target_clean:
                    indices.add(idx)
                    found_normalized = True

            if found_normalized:
                continue

            best_ratio = 0.0
            best_idx = -1

            for idx, source_sent in enumerate(source_sentences):
                matcher = difflib.SequenceMatcher(None, target, source_sent)
                if matcher.quick_ratio() > 0.8:
                    ratio = matcher.ratio()
                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_idx = idx

            if best_ratio > 0.85 and best_idx != -1:
                indices.add(best_idx)

        return sorted(list(indices))

    def _process_window(
        self, window_sentences: List[str], model: str, temperature: float
    ) -> List[Dict[str, Any]]:
        """
        Sends a window of text to the LLM for Named Entity Recognition and processes the response.
        Returns a list of raw dictionaries containing the entity, label, and resolved local indices.
        """
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
            "- MISC: Events, laws, products, works of art, nationalities.\n\n"
            "Task:\n"
            "Extract entities and group the EXACT sentences where they appear (including pronouns/references).\n\n"
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
                        {
                            "entity": item.entity,
                            "label": item.label,
                            "indices": local_indices,
                        }
                    )
        return results

    def extract_entities(
        self,
        chunks: List[ChunkPayload],
        window_size: int = 20,
        window_overlap: int = 5,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        **kwargs: Any,
    ) -> List[EntityChunkPayload]:
        """
        Orchestrates the Named Entity Recognition (NER) process over a sequence of text chunks.

        This method implements a sliding window strategy to process large lists of chunks
        while maintaining context continuity. It aggregates results from local windows
        into a global set of unique entities, resolving their occurrences back to the
        original physical chunks.

        The process consists of three main stages:
        1. **Pre-processing**: Extracts the raw text from the provided `ChunkPayload` objects
           and creates a positional mapping to allow O(1) retrieval of original chunks by index.
        2. **Sliding Window Execution**: Iterates through the text sentences in overlapping windows.
           For each window, it calls the LLM via `_process_window` to extract entities and their
           corresponding sentences. The overlap ensures that entities spanning across window
           boundaries are correctly captured.
        3. **Global Resolution**: Aggregates the local indices returned by the window processing
           into global indices relative to the input list. It then reconstructs the final
           `EntityChunkPayload` objects, populating their `children` field with the actual
           `ChunkPayload` references where the entity was found.

        Args:
            chunks (List[ChunkPayload]): The list of physical text segments (e.g., sentences)
                to be analyzed. Empty chunks are skipped.
            window_size (int): The number of chunks to include in a single LLM prompt context.
                Defaults to 20.
            window_overlap (int): The number of chunks that overlap between consecutive windows
                to preserve context. Must be strictly smaller than `window_size`. Defaults to 5.
            model (Optional[str]): The identifier of the specific LLM model to use for this
                operation. If not provided, the provider's default model is used.
            temperature (Optional[float]): The sampling temperature for the model. Lower values
                (e.g., 0.0) are recommended for extraction tasks to ensure determinism.
                If not provided, the provider's default is used.
            **kwargs: Additional keyword arguments passed to the underlying API call.

        Returns:
            List[EntityChunkPayload]: A list of unique entities found in the text. Each payload
            contains the entity name, its type (label), and a list of references to the
            original chunks (`children`) containing the evidence.

        Raises:
            ValueError: If `window_overlap` is greater than or equal to `window_size`.
        """
        if not chunks:
            return []

        sentences = [c.content for c in chunks if c.content]
        chunk_map = {i: c for i, c in enumerate(chunks) if c.content}

        if window_overlap >= window_size:
            raise ValueError("window_overlap must be strictly less than window_size")

        active_model = model or self.default_model
        active_temp = (
            temperature if temperature is not None else self.default_temperature
        )

        merged_entities: Dict[Tuple[str, str], Set[int]] = {}

        step = max(1, window_size - window_overlap)

        for i in range(0, len(sentences), step):
            window = sentences[i : i + window_size]

            if i > 0 and len(window) < window_overlap and len(sentences) > window_size:
                continue

            window_results = self._process_window(window, active_model, active_temp)

            for res in window_results:
                key = (res["entity"], res["label"])
                if key not in merged_entities:
                    merged_entities[key] = set()

                global_indices = {local_idx + i for local_idx in res["indices"]}
                merged_entities[key].update(global_indices)

            if i + window_size >= len(sentences):
                break

        final_payloads = []

        for (name, label), indices_set in merged_entities.items():
            child_chunks = []
            for idx in sorted(list(indices_set)):
                if idx in chunk_map:
                    child_chunks.append(chunk_map[idx])

            if child_chunks:
                payload = EntityChunkPayload(
                    entity=name, type=label, children=child_chunks, content=name
                )
                final_payloads.append(payload)

        return final_payloads

    def summarize(self, chunks: List[ChunkPayload], **kwargs: Any) -> List[ChunkPayload]:
        """
        Generates summaries for a list of chunks, employing batching to optimize 
        LLM calls. Returns new ChunkPayloads wrapping the summaries.
        """
        if not chunks:
            return []

        valid_indices = [i for i, c in enumerate(chunks) if c.content]
        valid_texts = [chunks[i].content for i in valid_indices]

        if not valid_texts:
            return []

        summaries_map: Dict[int, str] = {}
        batch_size = kwargs.get("batch_size", 5) 

        for i in range(0, len(valid_texts), batch_size):
            batch_texts = valid_texts[i : i + batch_size]
            batch_indices = valid_indices[i : i + batch_size]
            
            batch_summaries = self._summarize_batch(batch_texts, **kwargs)
            
            for local_idx, summary_text in enumerate(batch_summaries):
                if local_idx < len(batch_indices):
                    global_idx = batch_indices[local_idx]
                    summaries_map[global_idx] = summary_text

        results = []
        for i, original_chunk in enumerate(chunks):
            if i in summaries_map and summaries_map[i]:
                summary_text = summaries_map[i]
                
                payload = ChunkPayload(
                    content=summary_text,
                    content_raw=summary_text, 
                    children=[original_chunk], 
                )
                results.append(payload)
            else:
                pass

        return results

    def _summarize_batch(self, batch_texts: List[str], **kwargs) -> List[str]:
        """
        Internal helper to summarize multiple texts in one prompt.
        """
        results = [""] * len(batch_texts)
        
        formatted_targets = "\n\n".join([f"TEXT {i+1}:\n{t}" for i, t in enumerate(batch_texts)])
        
        system_prompt = (
            "You are an expert summarizer. Summarize each of the following texts independently.\n"
            "Return the output as a JSON list of strings, strictly preserving the order."
        )
        
        class BatchSummaryResponse(BaseModel):
            summaries: List[str]

        try:
            completion = self.client.beta.chat.completions.parse(
                model=kwargs.get("model", self.default_model),
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": formatted_targets}
                ],
                response_format=BatchSummaryResponse,
                temperature=0.2,
            )
            
            if completion.choices[0].message.parsed:
                returned_sums = completion.choices[0].message.parsed.summaries
                for i in range(min(len(returned_sums), len(results))):
                    results[i] = returned_sums[i]
                    
        except Exception as e:
            self.logger.error(f"Batch summary failed: {e}")
            pass
            
        return results
    
    def assign_topics(self, chunks: List[ChunkPayload], **kwargs: Any) -> List[TopicChunkPayload]:
        """
        Extracts topics from the provided chunks and groups them semantically.
        Implements the pivot logic: transforms a list of chunks with topics into
        a list of topics containing chunks.
        """
        if not chunks:
            return []

        valid_chunks_with_indices = [(i, c) for i, c in enumerate(chunks) if c.content]
        valid_texts = [c.content for _, c in valid_chunks_with_indices]
        
        if not valid_texts:
            return []

        taxonomy = kwargs.get("taxonomy")
        if not taxonomy:
            taxonomy = self._discover_taxonomy(valid_texts, **kwargs)

        chunk_topic_map: Dict[int, List[str]] = {}
        batch_size = kwargs.get("batch_size", 10)
        
        for i in range(0, len(valid_texts), batch_size):
            batch_texts = valid_texts[i : i + batch_size]
            batch_start_index = i 
            
            assignments = self._classify_batch_topics(batch_texts, taxonomy, **kwargs)
            
            for local_idx, topics in assignments.items():
                global_idx = batch_start_index + local_idx
                chunk_topic_map[global_idx] = topics

        topic_grouping: Dict[str, List[ChunkPayload]] = {}

        for global_idx, topics in chunk_topic_map.items():
            original_chunk = valid_chunks_with_indices[global_idx][1]
            
            for topic in topics:
                if topic not in topic_grouping:
                    topic_grouping[topic] = []
                
                if original_chunk not in topic_grouping[topic]:
                    topic_grouping[topic].append(original_chunk)

        final_payloads = []
        for topic_label, grouped_chunks in topic_grouping.items():
            payload = TopicChunkPayload(
                content=topic_label,
                content_raw="", 
                topic=topic_label,
                children=grouped_chunks
            )
            final_payloads.append(payload)

        return final_payloads

    def _discover_taxonomy(self, texts: List[str], **kwargs: Any) -> List[str]:
        """Generates a taxonomy from the text content."""
        batch_size = kwargs.get("taxonomy_batch_size", 10)
        all_raw_topics = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            prompt = "Extract distinct key topics from the text. Output as list."
            try:
                response = self.client.chat.completions.create(
                    model=kwargs.get("model", self.default_model),
                    messages=[{"role": "system", "content": prompt}, {"role": "user", "content": "\n".join(batch)}],
                    temperature=0.3
                )
                content = response.choices[0].message.content
                if content:
                    all_raw_topics.extend([line.strip("- *") for line in content.split('\n') if line.strip()])
            except Exception:
                continue
        
        unification_prompt = "Consolidate into a clean, deduplicated taxonomy."
        try:
            completion = self.client.beta.chat.completions.parse(
                model=kwargs.get("model", self.default_model),
                messages=[{"role": "system", "content": unification_prompt}, {"role": "user", "content": "\n".join(all_raw_topics[:2000])}],
                response_format=LLMTaxonomyResponse,
            )
            return completion.choices[0].message.parsed.topics
        except Exception:
            return []

    def _classify_batch_topics(self, batch_texts: List[str], taxonomy: List[str], **kwargs: Any) -> Dict[int, List[str]]:
        """
        Classifies a batch of texts against the taxonomy. 
        Returns a map of local_batch_index -> topics.
        """
        results = {}
        formatted_batch = "\n\n--- BLOCK ---\n".join(batch_texts)
        prompt = f"Taxonomy: {taxonomy}\nAssign topics. Return 'anchor' (first 10 words) for matching."
        
        try:
            completion = self.client.beta.chat.completions.parse(
                model=kwargs.get("model", self.default_model),
                messages=[{"role": "system", "content": prompt}, {"role": "user", "content": formatted_batch}],
                response_format=LLMTopicResponse,
                temperature=0.0,
            )
            
            if completion.choices[0].message.parsed:
                assignments = completion.choices[0].message.parsed.assignments
                
                available_indices = set(range(len(batch_texts)))
                
                for asm in assignments:
                    target_anchor = asm.anchor.strip().lower()
                    best_idx = -1
                    best_score = 0.0
                    
                    for idx in available_indices:
                        chunk_start = batch_texts[idx][:len(asm.anchor) + 20].strip().lower()
                        
                        if chunk_start.startswith(target_anchor):
                            best_score = 1.0
                            best_idx = idx
                            break
                        
                        matcher = difflib.SequenceMatcher(None, target_anchor, chunk_start)
                        score = matcher.quick_ratio()
                        if score > best_score:
                            best_score = score
                            best_idx = idx
                            
                    if best_idx != -1 and best_score > 0.8:
                        results[best_idx] = asm.topics

        except Exception as e:
            self.logger.error(f"Error classifying topic batch: {e}")
            
        return results
    
    def _map_assignments_to_results(
        self,
        assignments: List[LLMTopicAssignment],
        current_batch: List[str],
        global_results: List[List[str]],
        global_offset: int,
    ) -> None:
        """
        Resolves the mapping between LLM responses and original text chunks using Fuzzy Logic.

        This helper method iterates through the assignments returned by the LLM
        and attempts to locate the corresponding chunk in the 'current_batch'.
        It employs a two-tier matching strategy:

        1. Prefix Matching: Checks if the chunk starts with the anchor text (case-insensitive).
           This covers the majority of cases where the LLM quotes accurately.

        2. Fuzzy Similarity (Fallback): Uses 'difflib.SequenceMatcher' to compare
           the anchor against the start of the chunk. This handles scenarios where
           the LLM might have introduced minor typos, punctuation changes, or
           spacing discrepancies in the anchor text.

        If a match is found with a confidence score above 0.8, the topics are
        assigned to the corresponding index in the 'global_results' list.

        Args:
            assignments: The list of topic assignments returned by the LLM.
            current_batch: The actual text strings of the current batch being processed.
            global_results: The master list of results to be updated in-place.
            global_offset: The starting index of the current batch in the global list.
        """
        available_indices = set(range(len(current_batch)))

        for asm in assignments:
            best_idx = -1
            best_score = 0.0

            target_anchor = asm.anchor.strip().lower()

            for idx in available_indices:
                chunk_start = current_batch[idx][: len(asm.anchor) + 20].strip().lower()

                if chunk_start.startswith(target_anchor):
                    best_score = 1.0
                    best_idx = idx
                    break

                matcher = difflib.SequenceMatcher(None, target_anchor, chunk_start)
                score = matcher.quick_ratio()
                if score > best_score:
                    best_score = score
                    best_idx = idx

            if best_idx != -1 and best_score > 0.8:
                global_index = global_offset + best_idx
                global_results[global_index] = asm.topics

    def assign_context(self, chunks: List[ChunkPayload], **kwargs: Any) -> List[ContextualChunkPayload]:
        """
        Generates situational context for each chunk and wraps it in a ContextualChunkPayload.
        Uses _generate_context_for_batch to handle the LLM interaction.
        """
        if not chunks:
            return []
            
        valid_indices = [i for i, c in enumerate(chunks) if c.content]
        valid_texts = [chunks[i].content for i in valid_indices]
        
        if not valid_texts:
            return []

        contexts_map: Dict[int, str] = {}
        batch_size = kwargs.get("batch_size", 5)
        
        for i in range(0, len(valid_texts), batch_size):
            batch_texts = valid_texts[i : i + batch_size]
            batch_indices = valid_indices[i : i + batch_size]
            
            batch_contexts = self._generate_context_for_batch(batch_texts, valid_texts, i, **kwargs)
            
            for local_idx, ctx_str in enumerate(batch_contexts):
                if local_idx < len(batch_indices) and ctx_str:
                    global_idx = batch_indices[local_idx]
                    contexts_map[global_idx] = ctx_str

        results = []
        for i, original_chunk in enumerate(chunks):
            ctx_str = contexts_map.get(i, "")
            
            payload = ContextualChunkPayload(
                content=original_chunk.content,
                content_raw=original_chunk.content_raw,
                context=ctx_str,
                children=[original_chunk], 
                annotations=original_chunk.annotations.copy()
            )
            results.append(payload)
                
        return results

    def _generate_context_for_batch(
        self, 
        batch_texts: List[str], 
        all_texts: List[str], 
        offset: int, 
        **kwargs
    ) -> List[str]:
        """
        New helper method: Encapsulates the logic of calling the LLM and matching
        the returned quotes to the original text using your robust token overlap strategy.
        """
        overlap_size = kwargs.get("overlap_size", 3)
        start_overlap = max(0, offset - overlap_size)
        
        prev_chunks = all_texts[start_overlap : offset]
        background_text = "\n---\n".join(prev_chunks) if prev_chunks else "start of document."
        targets_text = "\n\n".join([f"TARGET FRAGMENT:\n{t}" for t in batch_texts])

        system_prompt = (
            "You are an expert Knowledge Graph Engineer specializing in Contextual Retrieval.\n\n"
            "GOAL: Provide situational context for isolated fragments.\n"
            "1. IDENTITY: Resolve pronouns using names from the background.\n"
            "2. LOCATION: Anchor the fragment to its document section or parent topic.\n"
            "3. CAUSALITY: Explain the event in the background that leads to this fragment.\n\n"
            "CONSTRAINTS:\n"
            "- DO NOT summarize the target fragment. Context must be purely ADDITIVE.\n"
            "- Use the BACKGROUND as the source of truth.\n"
            "- Return a JSON mapping for each fragment using its starting words as 'quote'."
        )

        user_prompt = (
            f"<BACKGROUND (READ ONLY)>\n{background_text}\n</BACKGROUND>\n\n"
            f"<TARGETS (PROCESS THESE)>\n{targets_text}\n</TARGETS>"
        )

        results = [""] * len(batch_texts)

        try:
            completion = self.client.beta.chat.completions.parse(
                model=kwargs.get("model", self.default_model),
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format=LLMContextResponse,
                temperature=0.0,
            )

            if completion.choices[0].message.parsed:
                def get_tokens(text: str) -> set:
                    return set("".join(filter(str.isalnum, word)).lower() for word in text.split())

                batch_token_sets = [
                    (local_idx, get_tokens(chunk[:50])) 
                    for local_idx, chunk in enumerate(batch_texts)
                ]

                for assignment in completion.choices[0].message.parsed.assignments:
                    assignment_tokens = get_tokens(assignment.quote)
                    if not assignment_tokens:
                        continue

                    best_match_idx = -1
                    max_overlap = 0

                    for local_idx, chunk_tokens in batch_token_sets:
                        overlap = len(assignment_tokens.intersection(chunk_tokens))
                        if overlap > max_overlap:
                            max_overlap = overlap
                            best_match_idx = local_idx
                    
                    if best_match_idx != -1 and max_overlap > 0:
                        results[best_match_idx] = assignment.context

        except Exception as e:
            self.logger.error(f"Context generation failed: {e}")
            
        return results
    
    def extract_zettels(
        self, chunks: List[ChunkPayload], batch_size: int = 5, **kwargs: Any
    ) -> List[ZettelChunkPayload]:
        """
        Implements the ZettelProviderProtocol.extract_zettels method.
        Processes chunks in batches with context propagation to generate interconnected Zettels.
        """
        if not chunks:
            return []

        global_handle_map: Dict[str, uuid.UUID] = {}
        raw_extraction_results: List[Dict[str, Any]] = []

        batches = [
            chunks[i : i + batch_size] for i in range(0, len(chunks), batch_size)
        ]

        for batch_index, batch in enumerate(batches):
            try:
                previous_concepts = list(global_handle_map.keys())
                system_prompt = self._build_zettel_system_prompt()
                user_prompt = self._build_zettel_user_prompt(batch, previous_concepts)

                completion = self.client.beta.chat.completions.parse(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    response_format=LLMZettelResponse,
                    **kwargs,
                )

                response_data = completion.choices[0].message.parsed

                if not response_data or not response_data.zettels:
                    continue

                for zettel_dto in response_data.zettels:
                    handle_key = zettel_dto.concept_handle.strip().lower()

                    if handle_key not in global_handle_map:
                        global_handle_map[handle_key] = uuid.uuid4()

                    current_uuid = global_handle_map[handle_key]

                    evidence_children = self._resolve_zettel_evidence(
                        zettel_dto.direct_quotes, batch
                    )

                    raw_extraction_results.append(
                        {
                            "uuid": current_uuid,
                            "dto": zettel_dto,
                            "children": evidence_children,
                        }
                    )

            except Exception as e:
                self.logger.error(
                    f"Error processing Zettel batch {batch_index}: {str(e)}"
                )
                continue

        return self._finalize_zettel_graph(raw_extraction_results, global_handle_map)

    def _build_zettel_system_prompt(self) -> str:
        """
        Constructs the system instructions specific to Zettelkasten extraction.
        """
        return (
            "You are an expert Knowledge Architect specialized in the Zettelkasten method. "
            "Your task is to decompose the provided text into atomic, interconnected notes (Zettels).\n\n"
            "CRITICAL RULES:\n"
            "1. ATOMICITY: Each Zettel must contain exactly ONE distinct idea or concept.\n"
            "2. INDEPENDENCE: The 'hypothesis' must be self-explanatory and fully understood without "
            "reading the original text.\n"
            "3. LANGUAGE: All generated content (hypothesis, titles, tags) MUST BE in the SAME LANGUAGE "
            "as the input text.\n"
            "4. EVIDENCE: You must copy exact text fragments into 'direct_quotes' to prove your hypothesis.\n"
            "5. CONNECTIVITY: Link your new notes to previously identified concepts (provided in context) "
            "or to other new notes in this batch using their 'concept_handle'."
        )

    def _build_zettel_user_prompt(
        self, batch: List[ChunkPayload], previous_concepts: List[str]
    ) -> str:
        """
        Constructs the user prompt for Zettel extraction with context injection.
        """
        formatted_context = ""
        if previous_concepts:
            formatted_context = (
                "### PREVIOUS KNOWLEDGE CONTEXT\n"
                "The following concepts have already been identified in previous sections. "
                "Use these names in 'related_concept_handles' if the new text relates to them:\n"
                f"- {', '.join(previous_concepts)}\n\n"
            )

        formatted_text_blocks = []
        for chunk in batch:
            content = chunk.content if chunk.content else ""
            formatted_text_blocks.append(f"--- BLOCK ---\n{content}")

        input_text = "\n".join(formatted_text_blocks)

        return (
            f"{formatted_context}"
            "### INPUT TEXT TO ANALYZE\n"
            f"{input_text}\n\n"
            "### TASK\n"
            "Extract atomic Zettels from the INPUT TEXT above. Ensure outputs are in the language of the text."
        )

    def _resolve_zettel_evidence(
        self, quotes: List[str], batch: List[ChunkPayload]
    ) -> List[ChunkPayload]:
        """
        Maps textual quotes provided by the LLM back to the original physical ChunkPayload objects
        using fuzzy matching.
        """
        matched_chunks: List[ChunkPayload] = []

        for quote in quotes:
            best_match = None
            best_score = 0.0

            quote_tokens = set(quote.lower().split())
            if not quote_tokens:
                continue

            for chunk in batch:
                if not chunk.content:
                    continue

                chunk_tokens = set(chunk.content.lower().split())
                if not chunk_tokens:
                    continue

                intersection = quote_tokens.intersection(chunk_tokens)
                union = quote_tokens.union(chunk_tokens)

                score = len(intersection) / len(union) if union else 0.0

                if score > best_score and score > 0.05:
                    best_score = score
                    best_match = chunk

            if best_score < 0.9:
                for chunk in batch:
                    if chunk.content and quote in chunk.content:
                        best_match = chunk
                        break

            if best_match and best_match not in matched_chunks:
                matched_chunks.append(best_match)

        return matched_chunks

    def _finalize_zettel_graph(
        self, raw_results: List[Dict[str, Any]], handle_map: Dict[str, uuid.UUID]
    ) -> List[ZettelChunkPayload]:
        """
        Resolves textual handle references into UUIDs and constructs the final payloads.
        """
        final_payloads = []

        for item in raw_results:
            dto: LLMZettelRef = item["dto"]
            current_uuid = item["uuid"]
            children = item["children"]

            resolved_related_ids: List[uuid.UUID] = []

            for handle in dto.related_concept_handles:
                clean_handle = handle.strip().lower()
                target_uuid = handle_map.get(clean_handle)

                if target_uuid and target_uuid != current_uuid:
                    if target_uuid not in resolved_related_ids:
                        resolved_related_ids.append(target_uuid)

            payload = ZettelChunkPayload(
                id=current_uuid,
                hypothesis=dto.hypothesis,
                tags=dto.tags,
                related_zettel_ids=resolved_related_ids,
                children=children,
                content=None,
            )
            final_payloads.append(payload)

        return final_payloads
