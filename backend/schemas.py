from pydantic import BaseModel
from datetime import datetime
from typing import Optional

class SessionBase(BaseModel):
    question: str
    transcript: str
    overall_score: float
    clarity_rating: float
    pace_rating: float
    visual_rating: float
    dominant_emotion: str

class SessionCreate(SessionBase):
    pass

class SessionResponse(SessionBase):
    id: int
    session_id: str
    timestamp: datetime
    star_compliance: bool
    summary: Optional[str] = None

    class Config:
        from_attributes = True
        