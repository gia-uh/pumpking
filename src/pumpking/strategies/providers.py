import os
import difflib
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


class LLMTopicAssignment(BaseModel):
    """
    Represents the assignment of semantic topics to a specific text block.

    This model enforces a content-based alignment strategy ("Anchoring") rather than
    relying on fragile numeric indexing. By requiring the LLM to quote the start
    of the text block, we can deterministically map the classification back to the
    original source chunk using fuzzy string matching, independent of the LLM's
    ability to count or maintain list order.
    """
    anchor: str = Field(
        ..., 
        description="The first 10 to 15 words of the text block exactly as they appear."
    )
    topics: List[str] = Field(
        ..., 
        description="The list of applicable topics derived from the provided taxonomy."
    )


class LLMTopicResponse(BaseModel):
    """
    Structured response wrapper for a batch classification request.
    """
    assignments: List[LLMTopicAssignment]


class LLMTaxonomyResponse(BaseModel):
    """
    Structured response for the taxonomy discovery phase.
    """
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
        """
        Maps LLM-returned sentences back to original indices using a tiered approach:
        1. Exact Match (O(1))
        2. Normalized Match (O(n))
        3. Fuzzy Match (O(n*m)) - For robustness against minor LLM alterations.
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

    def summarize(self, text: str, **kwargs: Any) -> str:
        """
        Generates a balanced, information-rich summary of the provided text.
        
        This implementation is opinionated: it aims for the 'sweet spot' between 
        high-level abstraction (gist) and factual preservation (extraction). 
        It is designed to produce a dense representation suitable for both 
        human reading and semantic embedding, ensuring that key entities 
        supporting the main narrative are retained.

        Constraints enforced via prompt:
        1. Strict adherence to the input language.
        2. Direct output without meta-commentary/fillers.
        3. Preservation of critical proper nouns and dates within the narrative flow.

        Args:
            text: The raw text to summarize.
            **kwargs: Configuration arguments (e.g., 'model').

        Returns:
            The summary string. If generation fails for any reason, returns 
            the original text to ensure data continuity.
        """
        if not text or not text.strip():
            return ""

        model = kwargs.get("model", self.default_model)
        # Temperature 0.2 provides stability without being overly repetitive
        temperature = kwargs.get("temperature", 0.2)

        system_prompt = (
            "You are an expert content summarizer. Produce a comprehensive yet concise "
            "summary of the provided text.\n\n"
            "EXECUTION GUIDELINES:\n"
            "1. BALANCE: Capture the core narrative/arguments while retaining key supporting "
            "details (specific names, dates, organizations). Do not over-generalize, "
            "but do not simply list facts.\n"
            "2. LANGUAGE: Output strictly in the SAME LANGUAGE as the input text.\n"
            "3. NO FILLERS: Start directly with the summary content. Do not use phrases "
            "like 'The text discusses' or 'In summary'.\n"
            "4. CLARITY: Remove redundancy and fluff."
        )

        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": text}
                ],
                temperature=temperature,
            )
            
            summary = response.choices[0].message.content
            return summary.strip() if summary else text

        except Exception:
            # Silent fallback: Return original text to prevent pipeline breakage.
            return text

    def assign_topics(self, chunks: List[str], **kwargs: Any) -> List[List[str]]:
        """
        Orchestrates the semantic classification of a list of text chunks.

        This method implements a robust, two-phase architectural pattern to handle
        topic assignment at scale:

        1. Taxonomy Discovery (Optional): If a pre-defined taxonomy is not provided
           in the keyword arguments, the method triggers a Map-Reduce process.
           It scans the entire document content in batches to extract raw themes
           and then consolidates them into a unified, deduplicated global taxonomy.
           This ensures that the topics are contextually relevant to the specific document.

        2. Batch Classification with Anchor Matching: It iterates through the input
           chunks in batches to optimize context window usage and API costs.
           Unlike traditional index-based approaches, it employs an "Anchor Matching"
           strategy. The LLM is required to quote the beginning of the text it is
           classifying. These quotes are then fuzzy-matched against the original
           chunks to guarantee 1-to-1 alignment, making the system resilient to
           LLM hallucinations or off-by-one indexing errors.

        Args:
            chunks: A list of text strings (paragraphs or sections) to be classified.
            **kwargs: Configuration parameters including:
                      - 'taxonomy' (List[str], optional): A pre-computed list of topics.
                      - 'taxonomy_batch_size' (int): Batch size for discovery (default: 10).
                      - 'batch_size' (int): Batch size for classification (default: 10).
                      - 'model' (str): The specific LLM model to use.

        Returns:
            A list of lists of strings, strictly aligned with the input 'chunks'.
            The element at index 'i' in the return list corresponds to the topics
            assigned to 'chunks[i]'.
        """
        taxonomy = kwargs.get("taxonomy")
        if not taxonomy:
            taxonomy = self._discover_taxonomy(chunks, **kwargs)
            
        return self._classify_batches(chunks, taxonomy, **kwargs)

    def _discover_taxonomy(self, chunks: List[str], **kwargs: Any) -> List[str]:
        """
        Executes a Map-Reduce algorithm to generate a global taxonomy from the document.

        This method addresses the "Context Window Explosion" problem common in
        large document processing. Instead of feeding the entire text to the LLM
        at once, it processes the document iteratively.

        Phase 1 (Map): The method iterates through the document in chunks determined
        by 'taxonomy_batch_size'. For each batch, it requests a raw list of key
        topics. The prompt explicitly instructs the LLM to preserve the original
        language of the document.

        Phase 2 (Reduce): All raw topics collected from the batches are aggregated.
        A final LLM call is made to consolidate, deduplicate, normalize, and
        refine this aggregate list into a coherent taxonomy structure, again strictly
        enforcing the original language.

        Args:
            chunks: The full list of document text segments.
            **kwargs: Configuration options, specifically 'taxonomy_batch_size'.

        Returns:
            A sanitized list of unique string topics representing the global themes
            of the document.
        """
        batch_size = kwargs.get("taxonomy_batch_size", 10)
        all_raw_topics = []

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            batch_text = "\n---\n".join(batch)
            
            prompt = (
                "Analyze the following text fragments. "
                "Extract a list of the key distinct topics discussed. "
                "CRITICAL INSTRUCTION: Output the topics in the SAME LANGUAGE "
                "as the input text. Do not translate them to English. "
                "Return ONLY the list of topics."
            )
            
            response = self.client.chat.completions.create(
                model=kwargs.get("model", self.default_model),
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": batch_text}
                ],
                temperature=0.3
            )
            content = response.choices[0].message.content
            if content:
                raw_lines = [line.strip("- *") for line in content.split('\n') if line.strip()]
                all_raw_topics.extend(raw_lines)

        unification_prompt = (
            "Consolidate the following list of raw topics into a clean, "
            "deduplicated taxonomy. Merge synonyms and keep it concise. "
            "CRITICAL: Maintain the taxonomy in the ORIGINAL LANGUAGE of the source topics. "
            "Do not translate."
        )
        
        completion = self.client.beta.chat.completions.parse(
            model=kwargs.get("model", self.default_model),
            messages=[
                {"role": "system", "content": unification_prompt},
                {"role": "user", "content": "\n".join(all_raw_topics[:2000])} 
            ],
            response_format=LLMTaxonomyResponse,
        )
        
        return completion.choices[0].message.parsed.topics

    def _classify_batches(
        self, 
        chunks: List[str], 
        taxonomy: List[str], 
        **kwargs: Any
    ) -> List[List[str]]:
        """
        Performs topic assignment on batches of text using Anchor-based alignment.

        This method processes the input list in segments defined by 'batch_size'.
        For each batch, it constructs a prompt containing multiple visually separated
        text blocks.

        Crucially, it mandates that the LLM returns an 'anchor' string for each
        classification. This anchor is a quote of the first few words of the block.
        The prompt strictly forbids translation of this anchor to ensure that
        fuzzy matching against the original text succeeds.

        Args:
            chunks: The list of text segments to classify.
            taxonomy: The global list of valid topics to assign.
            **kwargs: Configuration options like 'batch_size' and 'model'.

        Returns:
            A list of topic lists aligned with the input chunks.
        """
        batch_size = kwargs.get("batch_size", 10)
        results = [[] for _ in chunks] 

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            
            batch_text = "\n\n--- BLOCK ---\n".join(batch)
            
            prompt = (
                f"Taxonomy: {taxonomy}\n\n"
                "Task: Assign applicable topics to each text block.\n"
                "Constraint 1: For verification, you must return the first 10 words "
                "of the block in the 'anchor' field EXACTLY as they appear in the text. "
                "DO NOT TRANSLATE the anchor.\n"
                "Constraint 2: Output topics in the SAME LANGUAGE as the input text."
            )

            try:
                completion = self.client.beta.chat.completions.parse(
                    model=kwargs.get("model", self.default_model),
                    messages=[
                        {"role": "system", "content": prompt},
                        {"role": "user", "content": batch_text},
                    ],
                    response_format=LLMTopicResponse,
                    temperature=0.0,
                )
                
                if completion.choices[0].message.parsed:
                    assignments = completion.choices[0].message.parsed.assignments
                    self._map_assignments_to_results(assignments, batch, results, i)
                    
            except Exception:
                continue
                
        return results

    def _map_assignments_to_results(
        self, 
        assignments: List[LLMTopicAssignment], 
        current_batch: List[str], 
        global_results: List[List[str]], 
        global_offset: int
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
                chunk_start = current_batch[idx][:len(asm.anchor) + 20].strip().lower()
                
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
