import uuid
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, model_validator

class PumpkingBaseModel(BaseModel):
    def to_dict(self) -> Dict[str, Any]:
        data = self.model_dump(mode='json', exclude_none=True)
        return self._clean_empty(data)

    @classmethod
    def _clean_empty(cls, data: Any) -> Any:
        if isinstance(data, dict):
            return {
                k: v_clean
                for k, v in data.items()
                if (v_clean := cls._clean_empty(v)) not in (None, {}, [])
            }
        elif isinstance(data, list):
            return [
                v_clean
                for v in data
                if (v_clean := cls._clean_empty(v)) not in (None, {}, [])
            ]
        return data


class ChunkPayload(PumpkingBaseModel):
    content: str
    annotations: Dict[str, Any] = Field(default_factory=dict)


class ChunkNode(PumpkingBaseModel):
    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    parent_id: Optional[uuid.UUID] = None
    content: str
    content_raw: Optional[str] = None
    annotations: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode='after')
    def clean_content_raw(self) -> 'ChunkNode':
        if self.content_raw == self.content:
            self.content_raw = None
        return self


class DocumentRoot(PumpkingBaseModel):
    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    original_filename: Optional[str] = None
    children: List[ChunkNode] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)