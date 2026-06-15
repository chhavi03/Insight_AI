from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean
from datetime import datetime
from backend.database import Base

class SessionModel(Base):
    __tablename__ = "sessions"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String, unique=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    question = Column(String, nullable=False)
    transcript = Column(String, nullable=True)
    overall_score = Column(Float, default=0.0)
    clarity_rating = Column(Float, default=0.0)
    pace_rating = Column(Float, default=0.0)
    visual_rating = Column(Float, default=0.0)
    dominant_emotion = Column(String, default="neutral")
    star_compliance = Column(Boolean, default=False)
    summary = Column(String, nullable=True)