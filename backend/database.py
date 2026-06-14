import datetime
from sqlalchemy import create_engine, Column, Integer, Float, String, Boolean, DateTime, ForeignKey
from sqlalchemy.orm import declarative_base, sessionmaker, relationship

# 1. Establish the relative file database connection string
# This creates a local file named 'insightai.db' inside your backend directory automatically.
DATABASE_URL = "sqlite:///./backend/insightai.db"

# Create the low-level database communication engine
# connect_args={"check_same_thread": False} is required strictly for SQLite to allow multi-threaded async calls
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})

# Establish the short-lived transaction factory layer
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class that our individual database model schemas will inherit from
Base = declarative_base()

# 2. Define the structural Database Tables (Models)
class InterviewSession(Base):
    """
    Parent table recording individual user interview instances.
    """
    __tablename__ = "interview_sessions"

    id = Column(Integer, primary_key=True, index=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    question_text = Column(String, nullable=False)
    
    # Establish a clean, bidirectional relational bridge to the performance data table
    analytics = relationship("ResponseAnalytics", uselist=False, back_populates="session", cascade="all, delete-orphan")

class ResponseAnalytics(Base):
    """
    Child table archiving multi-modal scores and structural feedback.
    Tied permanently to an InterviewSession parent row via a foreign key constraint.
    """
    __tablename__ = "response_analytics"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, ForeignKey("interview_sessions.id", ondelete="CASCADE"), unique=True)
    
    # Textual telemetry outputs
    transcript = Column(String, nullable=False)
    overall_score = Column(Float, nullable=False)
    clarity_rating = Column(Float, nullable=False)
    pace_rating = Column(Float, nullable=False)
    visual_rating = Column(Float, nullable=False)
    dominant_emotion = Column(String, default="neutral")
    
    # STAR layout feedback blocks
    is_star_compliant = Column(Boolean, default=False)
    executive_summary = Column(String, nullable=True)

    # Bridge mapping point back to the parent class object
    session = relationship("InterviewSession", back_populates="analytics")

# 3. Database initialization utility function
def init_db():
    print("[DATABASE] Building tables and compiling database structural matrix...")
    # Instructs SQLAlchemy to look at all classes inheriting from 'Base' and physically build them in SQLite
    Base.metadata.create_all(bind=engine)
    print("[DATABASE] Database initialization sequence complete. Engine active.")

if __name__ == "__main__":
    # Standalone validation pass
    init_db()
    
    # Verify transaction lifecycles function flawlessly
    db = SessionLocal()
    try:
        print("[DATABASE] Testing write cycle transaction locks...")
        test_session = InterviewSession(question_text="Tell me about a time you solved a technical bug.")
        db.add(test_session)
        db.commit()
        print(f"[DATABASE] Verified. Sample session recorded successfully with tracking ID: {test_session.id}")
    finally:
        db.close()