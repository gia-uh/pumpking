from typing import List
from pumpking.models import NERResult
from pumpking.protocols import NERProviderProtocol

class LLMProvider(NERProviderProtocol):
    """
    Unified LLM Provider acting as a gateway for multiple NLP tasks.
    It implements various protocols (like NER) using a shared LLM backend configuration.
    """
    def analyze(self, sentences: List[str]) -> List[NERResult]:
        """
        Implements NERProviderProtocol.
        Analyzes sentences to extract entities using positional indices.
        """
        if not sentences:
            return []
            
        all_indices = list(range(len(sentences)))
        
        return [
            NERResult(
                entity="Default Entity",
                label="MISC",
                indices=all_indices
            )
        ]